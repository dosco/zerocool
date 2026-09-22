class Zerocool < Formula
  desc "Native C++23/Metal inference engine for Apple Silicon"
  homepage "https://github.com/dosco/zerocool"
  license "Apache-2.0"
  head "https://github.com/dosco/zerocool.git", branch: "master"

  depends_on "cmake" => :build
  depends_on arch: :arm64
  depends_on macos: :sequoia

  def install
    system "cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                    "-DZEROCOOL_BUILD_DIAGNOSTICS=OFF", *std_cmake_args
    system "cmake", "--build", "build", "--target", "zerocool", "--parallel"
    bin.install "build/bin/zerocool"
    pkgshare.install "models.lock.json", "mixed-models.lock.json", "scripts/qwen"
    doc.install "README.md", "THIRD_PARTY_NOTICES.md"
  end

  def caveats
    <<~EOS
      zerocool installs the engine only. The checkpoint is ~104GB and is
      prepared into ~100GB of records before first use:

        bash #{pkgshare}/qwen/download.sh
        python3 #{pkgshare}/qwen/verify_checkpoint.py
        python3 #{pkgshare}/qwen/prepare_storage.py --output ~/.zerocool/prepared/q4-records-v1

      Requires roughly 200GB of free disk.
    EOS
  end

  test do
    assert_match "ZeroCool", shell_output("#{bin}/zerocool 2>&1", 1)
  end
end
