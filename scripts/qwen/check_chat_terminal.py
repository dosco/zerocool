#!/usr/bin/env python3
"""PTY checks of the compiled TUI against a separately built test-only server."""
import argparse
import codecs
import ctypes
import hashlib
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import socket
import struct
import subprocess
import termios
import time
import urllib.request
import pyte


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',type=Path,default=Path('build/qwen/chat_fixture'))
    ap.add_argument('--native',type=Path,default=Path('build/qwen/bin/zerocool'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();binary=args.fixture.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    cases=[];footprints=[]
    class Usage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_ubyte*16)]+[(n,ctypes.c_uint64) for n in 'user system pkg interrupt pageins wired resident footprint start exit child_user child_system child_pkg child_interrupt child_pageins child_elapsed disk_read disk_write'.split()]
    lib=ctypes.CDLL('/usr/lib/libproc.dylib')

    class Terminal:
        def __init__(self,arguments,executable=binary):
            self.master,self.slave=pty.openpty();self.original=termios.tcgetattr(self.slave)
            fcntl.ioctl(self.slave,termios.TIOCSWINSZ,struct.pack('HHHH',32,120,0,0))
            self.screen=pyte.Screen(120,32);self.stream=pyte.Stream(self.screen);self.decoder=codecs.getincrementaldecoder('utf-8')('replace')
            self.raw=bytearray()
            self.process=subprocess.Popen([str(executable),'chat',*arguments],stdin=self.slave,stdout=self.slave,stderr=self.slave,start_new_session=True)
        def pump(self,seconds=.1):
            usage=Usage()
            if lib.proc_pid_rusage(self.process.pid,2,ctypes.byref(usage))==0:footprints.append(usage.footprint)
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if select.select([self.master],[],[],.02)[0]:
                    try:data=os.read(self.master,65536)
                    except OSError:break
                    if not data:break
                    self.raw+=data;self.stream.feed(self.decoder.decode(data))
        def text(self):return '\n'.join(self.screen.display)
        def wait(self,needle,timeout=8):
            end=time.monotonic()+timeout
            while time.monotonic()<end:
                self.pump()
                if needle in self.text():return
                if self.process.poll() is not None:break
            raise AssertionError(f'Missing {needle!r}:\n{self.text()}')
        def write(self,value):os.write(self.master,value)
        def close(self):
            if self.process.poll() is None:
                self.write(b'\x11')
                until=time.monotonic()+10
                while self.process.poll() is None and time.monotonic()<until:self.pump()
                if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=5)
            self.pump();actual=termios.tcgetattr(self.slave);expected=list(self.original)
            # macOS sets PENDIN when canonical mode is restored. It is kernel
            # pending-input state, not a terminal configuration left behind.
            actual[3]&=~termios.PENDIN;expected[3]&=~termios.PENDIN
            restored=actual==expected
            os.close(self.master);os.close(self.slave);return restored

    terminal=None;server=None
    try:
        terminal=Terminal([]);terminal.wait('ready ·')
        children=subprocess.check_output(['pgrep','-P',str(terminal.process.pid)],text=True).split();assert len(children)==1,children
        child=int(children[0])
        (out/'default-child-command.txt').write_text(subprocess.check_output(['ps','-ww','-o','command=','-p',str(child)],text=True))
        terminal.write(b'\x1b[200~hello\nsecond line\x04\x03\x1b[201~');terminal.pump(.5)
        assert 'assistant' not in terminal.text(),terminal.text()
        assert 'second line' in terminal.text(),terminal.text()
        terminal.write(b'\x04');terminal.wait('OK 🦉')
        fcntl.ioctl(terminal.slave,termios.TIOCSWINSZ,struct.pack('HHHH',24,85,0,0));os.kill(terminal.process.pid,signal.SIGWINCH);terminal.pump(.3)
        terminal.screen.resize(lines=24,columns=85)
        terminal.write(b'\x1bOQ');terminal.pump(.2) # F2
        terminal.write(b'/new\x04');terminal.wait('New conversation');terminal.wait('ready ·')
        terminal.write(b'slow\x04');terminal.wait('Starting…')
        terminal.write(b'\x1b[200~must not replace retry\x1b[201~');terminal.pump(.1)
        assert 'must not replace retry' not in terminal.text()
        terminal.write(b'\x03');terminal.wait('Stopped.');terminal.wait('ready ·')
        assert 'slow' in terminal.text()
        terminal.write(b'\x15hello again\x04');terminal.wait('OK 🦉')
        path=out/'transcript.json';terminal.write(('/save '+str(path)).encode()+b'\x04');terminal.wait('Transcript saved')
        transcript=json.loads(path.read_text());assert any(e['interrupted'] for e in transcript['entries'])
        terminal.write(('/save '+str(path)).encode()+b'\x04');terminal.wait('never overwritten')
        (out/'owned-screen.txt').write_text(terminal.text());(out/'owned-terminal.raw').write_bytes(terminal.raw)
        restored=terminal.close();terminal=None;assert restored,'terminal attributes were not restored'
        try:os.kill(child,0)
        except ProcessLookupError:pass
        else:raise AssertionError('owned engine was left alive')
        cases.append(dict(name='owned_chat_paste_cancel_export_resize_cleanup',passed=True))

        a,b=socket.socketpair()
        server=subprocess.Popen([str(binary),'serve','--port','0','--control-fd',str(b.fileno())],pass_fds=[b.fileno()],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        b.close();a.settimeout(5);url=json.loads(a.recv(4096).split(b'\n')[0])['endpoint']
        terminal=Terminal(['--connect',url]);terminal.wait('ready ·');assert terminal.close();terminal=None
        with urllib.request.urlopen(url+'/health') as f:assert json.load(f)['status']=='ready'
        assert server.poll() is None
        a.close();server.wait(timeout=5);server=None
        cases.append(dict(name='connected_chat_does_not_stop_server',passed=True))

        a,b=socket.socketpair()
        server=subprocess.Popen([str(binary),'serve','--port','0','--control-fd',str(b.fileno())],pass_fds=[b.fileno()],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        b.close();a.settimeout(5);url=json.loads(a.recv(4096).split(b'\n')[0])['endpoint']
        terminal=Terminal(['--connect',url]);terminal.wait('ready ·')
        a.close();server.wait(timeout=5)
        a,b=socket.socketpair()
        server=subprocess.Popen([str(binary),'serve','--port',url.rsplit(':',1)[1],'--control-fd',str(b.fileno())],pass_fds=[b.fileno()],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        b.close();a.settimeout(5);assert json.loads(a.recv(4096).split(b'\n')[0])['endpoint']==url
        terminal.wait('server instance changed');terminal.wait('disconnected ·')
        terminal.write(b'hello\x04');terminal.wait('Wait until the engine is ready.')
        assert 'OK 🦉' not in terminal.text();assert terminal.close();terminal=None
        with urllib.request.urlopen(url+'/health') as f:assert json.load(f)['status']=='ready'
        a.close();server.wait(timeout=5);server=None
        cases.append(dict(name='replacement_server_requires_explicit_reconnect',passed=True))

        terminal=Terminal(['--model','unused-test-assets']);terminal.wait('ready ·')
        child=int(subprocess.check_output(['pgrep','-P',str(terminal.process.pid)],text=True).strip())
        terminal.process.kill();terminal.process.wait();terminal.close();terminal=None
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            r=subprocess.run(['ps','-o','stat=','-p',str(child)],capture_output=True,text=True)
            if r.returncode or 'Z' in r.stdout:break
            time.sleep(.05)
        else:raise AssertionError('engine did not exit after parent loss')
        cases.append(dict(name='parent_loss_stops_owned_engine',passed=True))

        terminal=Terminal(['--model',str(out/'missing-model-assets')],args.native.resolve());terminal.wait('failed ·')
        child=int(subprocess.check_output(['pgrep','-P',str(terminal.process.pid)],text=True).strip())
        (out/'native-failure-screen.txt').write_text(terminal.text())
        assert terminal.close();terminal=None
        try:os.kill(child,0)
        except ProcessLookupError:pass
        else:raise AssertionError('failed native engine was left alive')
        cases.append(dict(name='native_startup_failure_visible_and_cleanup',passed=True))
    except Exception as e:
        cases.append(dict(name='terminal_check',passed=False,error=str(e)))
        if terminal:(out/'failure-screen.txt').write_text(terminal.text());(out/'failure.raw').write_bytes(terminal.raw)
    finally:
        if terminal:terminal.close()
        if server:server.terminate();server.wait(timeout=5)
    result=dict(fixture_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),peak_sampled_client_footprint_bytes=max(footprints,default=0),kind='model_free_chat_terminal_v1',complete=True,passed=all(c['passed'] for c in cases),cases=cases,
        limitations=['Uses the test-only executor; does not establish real model correctness or speed.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
