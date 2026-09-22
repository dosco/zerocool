class Zerocool < Formula
  desc "Native C++23/Metal inference engine for Apple Silicon"
  homepage "https://github.com/dosco/zerocool"
  license "Apache-2.0"
  head "https://github.com/dosco/zerocool.git", branch: "master"

  depends_on "cmake" => :build
  depends_on arch: :arm64
  depends_on macos: :sequoia

  # Homebrew traps CMake FetchContent, so the pinned dependencies are staged
  # here instead. The revisions match cmake/qwen.cmake exactly: the project
  # pins commits rather than tags so a moved tag cannot change the engine
  # without changing its recorded build identity.
  resource "json" do
    url "https://github.com/nlohmann/json.git",
        revision: "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03"
  end

  resource "minja" do
    url "https://github.com/google/minja.git",
        revision: "021c2293c187789ef13d56c6cfd89c9b134fd80f"
  end

  resource "ftxui" do
    url "https://github.com/ArthurSonzogni/FTXUI.git",
        revision: "f921fad208912747c17d129a8ef75ec7624b6eec"
  end

  def install
    deps = buildpath/"brew-deps"
    resources.each { |r| r.stage(deps/r.name) }

    system "cmake", "-S", ".", "-B", "build",
           "-DCMAKE_BUILD_TYPE=Release",
           "-DZEROCOOL_BUILD_DIAGNOSTICS=OFF",
           "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
           "-DFETCHCONTENT_SOURCE_DIR_JSON=#{deps}/json",
           "-DFETCHCONTENT_SOURCE_DIR_MINJA=#{deps}/minja",
           "-DFETCHCONTENT_SOURCE_DIR_FTXUI=#{deps}/ftxui",
           *std_cmake_args
    system "cmake", "--build", "build", "--target", "zerocool", "--parallel"

    bin.install "build/bin/zerocool"
    pkgshare.install "models.lock.json", "mixed-models.lock.json", "scripts/qwen"
    doc.install "README.md", "THIRD_PARTY_NOTICES.md"
  end

  def caveats
    <<~EOS
      This installs the engine only. The checkpoint is about 104GB and is
      prepared into roughly 100GB of records before first use.

      Fetching and verifying need nothing else installed; the transfer resumes
      if it is interrupted:

        zerocool download

      Preparation still uses the bundled Python tooling:

        python3 #{opt_pkgshare}/qwen/prepare_storage.py \\
          --output ~/.zerocool/prepared/q4-records-v1

      Around 200GB of free disk is needed. The engine targets a 32GiB machine
      and holds a bounded working set of at most 22GiB.
    EOS
  end

  test do
    # Bare invocation prints usage and exits 0; an unknown subcommand exits 1.
    assert_match "ZeroCool", shell_output("#{bin}/zerocool")
    assert_match "zerocool:", shell_output("#{bin}/zerocool nonsense 2>&1", 1)
  end
end
