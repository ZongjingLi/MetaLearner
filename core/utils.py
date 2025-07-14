from termcolor import colored

class Colors:
    ICY_BLUE = '\033[38;5;39m'    # For keys and containers
    FROST = '\033[38;5;45m'       # For strings
    SARONITE = '\033[38;5;240m'   # For brackets and structure
    PALE_BLUE = '\033[38;5;153m'  # For values
    DEATH_KNIGHT = '\033[38;5;63m' # For numpy/torch objects
    LICH_PURPLE = '\033[38;5;135m' # For special values and dataclasses
    SCOURGE_GREEN = '\033[38;5;77m' # For types and metadata
    FROSTFIRE = '\033[38;5;201m'  # For requires_grad=True
    BLOOD = '\033[38;5;160m'      # For requires_grad=False
    RUNIC = '\033[38;5;51m'       # For dataclass field names
    BOLD = '\033[1m'
    RESET = '\033[0m'

def stmetaphors(metaphors) -> str:
    output = f"infering from {len(metaphors)} mismatch expressions\n"
    for rewrites in metaphors:
        if len(rewrites) == 0 : output += "expression literally makes sense, no metaphors detected.\n"
        if len(rewrites) == 2:  output += "expression suggest an extention, but no short cut inferred.\n"
        ### len == 2 suggests only extended version and origonal version is possible.
        else: output += "expression suggest extention and potential rewrite short cuts.\n"

        if len(rewrites) > 0: ### show the extention of the current concept.
            extend = rewrites[0]
            fname, domain = extend[0].split(":")
            fname = colored(fname,"cyan")
            s_domain = colored(domain,"blue", attrs=["bold"])
            stype = [str(t) for t in extend[1]]
            ttype = [str(t) for t in extend[2]]
            output += f"{fname} \
of domain {s_domain} defined on \
{Colors.SARONITE}{stype}{Colors.RESET} might be extended to \
{Colors.SARONITE}{ttype}{Colors.RESET}\n"

        if len(rewrites) > 2: ### show the short cut created
            short_cuts = [
            rewrite[0].split(":")[0] for rewrite in rewrites[2:]]
            _, domain = rewrites[-1][0].split(":")
            t_domain = colored(domain, "blue", attrs = ["bold"])

            output += f"{colored(short_cuts,'cyan')}\n of domain {t_domain} defined on \
{Colors.SARONITE}{ttype}{Colors.RESET} might be rewrite to {fname} of domain {s_domain} defined on \
{Colors.SARONITE}{stype}{Colors.RESET}\n"
        output += "\n"
    return output


