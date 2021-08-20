{ pkgs ? import <nixpkgs> {} }:

let
  pyPkgs = pkgs.python38Packages;
in
  pyPkgs.buildPythonApplication {
    pname = "search-py";
    version = "0.0.0";

    src = pkgs.lib.cleanSource ./.;

    # Build environment
    buildInputs = with pyPkgs; [
      importmagic
      epc
      python-language-server
      graphviz
    ];
    # Run
    propagatedBuildInputs = with pyPkgs; [
      numpy
      pygame_sdl2
      termcolor
    ];
    # Check
    checkInputs = with pyPkgs; [
      autopep8
      mypy
      pylint
      pytest
    ];
    checkPhase = ''
      autopep8 --aggressive --exit-code */.py **/.py
      pytest
      mypy ./main.py
    '';

    meta = {
      homepage = "https://github.com/Dietr1ch/search-py";
      license = pkgs.lib.licenses.bsd3;
      description = "A library with search algorithms focused on teaching.";
    };
  }
