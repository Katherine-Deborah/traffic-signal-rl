"""
Generate SUMO network files for the traffic RL project.

Usage:
    python network/generate_network.py          # single intersection (default)
    python network/generate_network.py --grid   # 3x3 grid (Phase 6)

Outputs (network/single/):
    single.nod.xml          node definitions
    single.edg.xml          edge definitions
    single.net.xml          compiled network (from netconvert)
    routes_normal.rou.xml   standard traffic flows
    routes_ns_peak.rou.xml  morning peak (NS heavy)
    routes_ew_peak.rou.xml  evening peak (EW heavy)
    routes_high.rou.xml     high-density scenario
    single.sumocfg          simulation config
"""

import argparse
import os
import subprocess
import sys

import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
#  Single intersection network
# ──────────────────────────────────────────────

def write_node_file(path: str) -> None:
    content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">

    <!-- Central traffic-light intersection -->
    <node id="center" x="0.0"    y="0.0"    type="traffic_light"/>

    <!-- Dead-end approach nodes, 300 m out in each direction -->
    <node id="north"  x="0.0"    y="300.0"  type="dead_end"/>
    <node id="south"  x="0.0"    y="-300.0" type="dead_end"/>
    <node id="east"   x="300.0"  y="0.0"    type="dead_end"/>
    <node id="west"   x="-300.0" y="0.0"    type="dead_end"/>

</nodes>
"""
    _write(path, content)


def write_edge_file(path: str, num_lanes: int = 2) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">

    <!-- Incoming edges (approach to intersection) -->
    <edge id="N2C" from="north"  to="center" numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="S2C" from="south"  to="center" numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="E2C" from="east"   to="center" numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="W2C" from="west"   to="center" numLanes="{num_lanes}" speed="13.89" priority="1"/>

    <!-- Outgoing edges (departure from intersection) -->
    <edge id="C2N" from="center" to="north"  numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="C2S" from="center" to="south"  numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="C2E" from="center" to="east"   numLanes="{num_lanes}" speed="13.89" priority="1"/>
    <edge id="C2W" from="center" to="west"   numLanes="{num_lanes}" speed="13.89" priority="1"/>

</edges>
"""
    _write(path, content)


def write_route_file(path: str, scenario: str = "normal") -> None:
    """
    Traffic flows (vehicles/hour) for 12 origin-destination pairs.
    Each pair covers the 4 through movements + 8 turn movements.
    """
    scenarios = {
        "normal": {
            "N2S": 300, "S2N": 300, "E2W": 300, "W2E": 300,
            "N2E": 100, "N2W": 100, "S2E": 100, "S2W": 100,
            "E2N": 100, "E2S": 100, "W2N": 100, "W2S": 100,
        },
        "ns_peak": {
            "N2S": 600, "S2N": 600, "E2W": 200, "W2E": 200,
            "N2E": 150, "N2W": 150, "S2E": 150, "S2W": 150,
            "E2N":  50, "E2S":  50, "W2N":  50, "W2S":  50,
        },
        "ew_peak": {
            "N2S": 200, "S2N": 200, "E2W": 600, "W2E": 600,
            "N2E":  50, "N2W":  50, "S2E":  50, "S2W":  50,
            "E2N": 150, "E2S": 150, "W2N": 150, "W2S": 150,
        },
        "high": {
            "N2S": 800, "S2N": 800, "E2W": 800, "W2E": 800,
            "N2E": 200, "N2W": 200, "S2E": 200, "S2W": 200,
            "E2N": 200, "E2S": 200, "W2N": 200, "W2S": 200,
        },
    }

    # Route (edge sequence) for each OD pair
    route_edges = {
        "N2S": "N2C C2S", "S2N": "S2C C2N",
        "E2W": "E2C C2W", "W2E": "W2C C2E",
        "N2E": "N2C C2E", "N2W": "N2C C2W",
        "S2E": "S2C C2E", "S2W": "S2C C2W",
        "E2N": "E2C C2N", "E2S": "E2C C2S",
        "W2N": "W2C C2N", "W2S": "W2C C2S",
    }

    flows = scenarios.get(scenario, scenarios["normal"])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" speedDev="0.1"',
        '           length="5.0" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>',
        '',
        '    <!-- Route definitions -->',
    ]

    for route_id, edges in route_edges.items():
        lines.append(f'    <route id="{route_id}" edges="{edges}"/>')

    lines.append('')
    lines.append('    <!-- Traffic flows -->')

    # Poisson arrivals: period="exp(rate)" draws exponentially distributed
    # insertion gaps (rate in veh/s), so vehicle arrival patterns vary with
    # SUMO's --seed. Plain vehsPerHour would insert equally spaced vehicles,
    # making every episode identical regardless of seed.
    for route_id, vph in flows.items():
        if vph > 0:
            rate = vph / 3600.0
            lines.append(
                f'    <flow id="flow_{route_id}" route="{route_id}" '
                f'begin="0" end="3600" period="exp({rate:.6f})" type="car"/>'
            )

    lines.append('</routes>')
    _write(path, "\n".join(lines))


def write_sumocfg(path: str, net_file: str, route_file: str) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>

    <report>
        <no-step-log value="true"/>
        <no-warnings value="false"/>
        <duration-log.statistics value="true"/>
    </report>

    <processing>
        <waiting-time-memory value="3600"/>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>

</configuration>
"""
    _write(path, content)


def run_netconvert(sumo_home: str, node_file: str, edge_file: str, output_file: str) -> None:
    netconvert = os.path.join(sumo_home, "bin", "netconvert.exe")
    if not os.path.exists(netconvert):
        netconvert = "netconvert"

    cmd = [
        netconvert,
        "--node-files", node_file,
        "--edge-files", edge_file,
        "--output-file", output_file,
        "--no-warnings",
        "--no-turnarounds",   # cleaner intersection, no U-turns
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[netconvert stderr]\n{result.stderr}")
        raise RuntimeError("netconvert failed. Check SUMO installation and paths.")

    print(f"  Created: {output_file}")


def print_tl_phases(sumo_home: str, net_file: str) -> None:
    """Read generated network and print traffic light phases for verification."""
    try:
        tools_path = os.path.join(sumo_home, "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        import sumolib
        net = sumolib.net.readNet(net_file)
        print("\n── Traffic Light Phases (auto-generated) ──")
        for tl in net.getTrafficLights():
            print(f"  TL id : {tl.getID()}")
            for prog in tl.getPrograms().values():
                print(f"  Prog  : {prog.getID()}")
                for i, phase in enumerate(prog.getPhases()):
                    tag = "GREEN" if "G" in phase.state and "y" not in phase.state else "      "
                    print(f"    Phase {i} [{tag}]: {phase.state}  (dur={phase.duration}s)")
    except Exception as exc:
        print(f"  (Could not read phases: {exc})")


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {path}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def generate_single(config: dict) -> None:
    sumo_home = config["sumo"]["home"]
    out_dir = "network/single"
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Generating single-intersection network ===")

    node_file = os.path.join(out_dir, "single.nod.xml")
    edge_file = os.path.join(out_dir, "single.edg.xml")
    net_file  = os.path.join(out_dir, "single.net.xml")

    write_node_file(node_file)
    write_edge_file(edge_file, num_lanes=2)
    run_netconvert(sumo_home, node_file, edge_file, net_file)

    print("\n── Generating route files ──")
    for scenario in ["normal", "ns_peak", "ew_peak", "high"]:
        route_file = os.path.join(out_dir, f"routes_{scenario}.rou.xml")
        write_route_file(route_file, scenario)

    print("\n── Generating sumocfg ──")
    write_sumocfg(
        os.path.join(out_dir, "single.sumocfg"),
        "single.net.xml",
        "routes_normal.rou.xml",
    )

    print_tl_phases(sumo_home, net_file)
    print(f"\n✓ Single-intersection network ready in '{out_dir}/'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SUMO network files")
    parser.add_argument("--grid", action="store_true", help="Generate 3x3 grid (Phase 6)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.grid:
        print("Grid network generation is implemented in Phase 6.")
        print("Run without --grid to generate the single-intersection network.")
    else:
        generate_single(config)


if __name__ == "__main__":
    main()
