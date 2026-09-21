// Test-only executable. Never installed or selectable by the production engine.
#include "engine/chat.hpp"
#include "chat_test_executor.hpp"
#include <csignal>
#include <print>
using namespace zerocool::engine;
namespace {std::atomic<bool> stop=false;void interrupt(int){stop=true;}}
int main(int argc,char** argv) {
    try {
        if(argc>1 && std::string(argv[1])=="chat")return chat_main(argc,argv);
        int port=0,fd=-1;
        for(int i=2;i+1<argc;i+=2) {std::string key=argv[i],value=argv[i+1];if(key=="--port")port=std::stoi(value);if(key=="--control-fd")fd=std::stoi(value);}
        std::signal(SIGTERM,interrupt);std::signal(SIGINT,interrupt);
        auto control=std::make_shared<chat_test::Control>();
        serve(Options{},uint16_t(port),&stop,fd,[&]{return std::make_unique<chat_test::TestExecutor>(control);});
    } catch(const std::exception& error) {std::println(stderr,"{}",error.what());return 1;}
}
