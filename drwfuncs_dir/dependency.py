"""File install all necessary dependencies."""
import sys
import subprocess
import pkg_resources


def install_dependencies():
    """Function wonder if some of dependency is missing, if so offers user to install them.
    If user denied, function print repositories where dependency can be found."""
    # define missing modules
    required = {'numpy': "https://www.numpy.org"}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required.keys() - installed
    if missing:
        # offer to install
        repos = {pkg: required[pkg] for pkg in missing}
        print("There are some missing dependency that is required to program work properly.\n"
              "  Would installed:")
        for module in missing:
            print(f"    {module};")
        # ask for respond
        to_install = input("Proceed (y/n)? ")
        # if wrong respond
        while to_install not in ("y", "n"):
            print(f"Your response ('{to_install}') was not one of the expected responses: y, n")
            to_install = input("Proceed (y/n)? ")
        # try to install them
        if to_install == "y":
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing])
            print()
            os.execl(python, python, *sys.argv)
            print()
        # give information to do it manually
        else:
            print("To make program work, you have to install:")
            for name, repo in repos.items():
                print(f"\tModule {name}, from {repo};")
            # exit program if modules where not installed
            exit(-1)
