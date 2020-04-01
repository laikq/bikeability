# Call this file with `nix-shell`
# There must be a nixos-unstable channel present,
# in case there is none add none via
# sudo nix-channel --add https://nixos.org/channels/nixos-unstable nixos-unstable

# nix-shell will throw you in a shell (bash) where python3 (command) is
# available. This python interpreter will have all the necessary packages
# installed to run bikeability.

# Make sure library-paths.patch is available in your current directory!

with import <nixos-unstable> {};

let
  py = python3;
  networkit = py.pkgs.buildPythonPackage rec {
    pname = "networkit";
    version = "5.0.1";
    src = py.pkgs.fetchPypi {
      inherit pname version;
      sha256 = "0f8v8dgqbjh7mr3q0z104hvfq25sr9dj0d41bxij2p49wykli77m";
      #      sha256 = "14zhz1nbqw9z3p433mbqfly03ph084qzj3y1hlzjba36xh6vbz7p";
    };
    nativeBuildInputs = [ cmake ninja py.pkgs.cython ];
    dontUseCmakeConfigure = true;  # cmake is called by setup.py
    doCheck = false;  # test input files not found
    propagatedBuildInputs = with py.pkgs; [
      tkinter
      sklearn-deap
      matplotlib
      seaborn
      pandas
      networkx
      tabulate
      ipython
    ];
  };
in
mkShell {
  buildInputs = with py.pkgs; [
    pip
    numpy
    scipy
    matplotlib
    networkit
    networkx
    virtualenv
    Rtree
  ];

  shellHook = ''
    # set SOURCE_DATE_EPOCH so that we can use python wheels
    SOURCE_DATE_EPOCH=$(date +%s)
    virtualenv --no-setuptools venv
    export PATH=$PWD/venv/bin:$PATH
    pip install -r requirements.txt
  '';
}
