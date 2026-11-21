{
    description = "AI specialization Python/Jupyter environment";

    inputs = {
      nixpkgs.url = "github:NixOS/nixpkgs/767542707d394ff15ac1981e903e005ba69528b5";
    };

    outputs = { self, nixpkgs }:
      let
        system = "x86_64-linux";

        pkgs = import nixpkgs {
          inherit system;
        };

        pythonEnv = pkgs.python310.withPackages (ps: with ps; [
          jupyterlab
          # notebook and nbclassic share commands in this nixpkgs revision.
          (pkgs.lib.hiPrio notebook)
          ipykernel
          jupytext

          scipy
          matplotlib
          sympy

          pandas
          requests
          scikit-learn

        ]);

        jupyterKernel = pkgs.runCommand "stasc-jupyter-kernel" {} ''
          mkdir -p $out/share/jupyter/kernels/stasc
          cat > $out/share/jupyter/kernels/stasc/kernel.json <<EOF
          ${builtins.toJSON {
            argv = [ "${pythonEnv}/bin/python" "-m" "ipykernel_launcher" "-f" "{connection_file}" ];
            display_name = "Python (STASC Nix)";
            language = "python";
          }}
          EOF
        '';
      in
      {
        devShells.${system}.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.jupyter
          ];

          shellHook = ''
            export JUPYTER_PATH="${jupyterKernel}/share/jupyter''${JUPYTER_PATH:+:$JUPYTER_PATH}"
            echo "AI specialization environment loaded"
            echo "Run: jupyter lab"
          '';
        };
      };
  }
