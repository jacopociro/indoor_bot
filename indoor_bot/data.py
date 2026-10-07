import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Genera i risultati per tutti gli esperimenti "
        "delle configurazioni config1/config2/config3."
    )
)

parser.add_argument(
    "--base-path",
    default="/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot",
    help="Cartella che contiene config1, config2 e config3",
)

parser.add_argument(
    "--ral",
    action="store_true",
    help="Use RAL/IEEE paper plot formatting",
)

parser.add_argument(
    "--noshow",
    action="store_true",
    help="Non mostrare le figure a schermo",
)

args = parser.parse_args()

BASE_PATH = args.base_path

CONFIGURATIONS = [
    "config1",
    "config2",
    "config3",
]


# ============================================================
# WAYPOINTS PER CONFIGURAZIONE
# ============================================================

WAYPOINTS = {

    # --------------------------------------------------------
    # CONFIG 1
    # --------------------------------------------------------
    "config1": np.array([
        [-6.5, -1.0],
        [ 3.0,  2.0],
        [ 3.5, -3.5],
        [-10.0, 2.0],
        [ 4.0, -2.5],
    ]),

    # --------------------------------------------------------
    # CONFIG 2
    # --------------------------------------------------------
    "config2": np.array([
        [-3.0,  0.0],
        [ 5.0,  2.5],
        [ 2.5, -4.5],
        [-10.0, 0.0],
        [ 5.0, -1.5],
    ]),

    # --------------------------------------------------------
    # CONFIG 3
    # --------------------------------------------------------
    "config3": np.array([
        [-5.0,  2.0],
        [ 4.0,  2.0],
        [ 0.0, -3.0],
        [-9.0,  3.0],
        [-8.0, -1.0],
    ]),
}


# ============================================================
# RAL / IEEE PLOT STYLE
# ============================================================

if args.ral:

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 0.1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    FIGSIZE = (4.0, 3.0)

else:

    FIGSIZE = None


# ============================================================
# FILES
# ============================================================

POSITION_FILES = [
    "rosbot_1_0_position.csv",
    "rosbot_2_1_position.csv",
    "rosbot_3_2_position.csv",
]

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]

SAVE_PDF = False

WAYPOINT_RADIUS = 2.5

HEADING_THRESHOLD_DEG = 45


# ============================================================
# UTILITY
# ============================================================

def wrap_angle(angle):

    return (
        (angle + np.pi) % (2 * np.pi)
        - np.pi
    )


def count_waypoint_visits(
    x,
    y,
    waypoints,
    threshold,
):
    """
    Conta il numero di ingressi nella regione di visita
    di ciascun waypoint.

    Una visita viene conteggiata quando il robot passa da:

        fuori dal waypoint -> dentro il waypoint

    Una permanenza consecutiva nella regione viene quindi
    conteggiata come una sola visita.

    Parameters
    ----------
    x : array-like
        Coordinate x del robot.

    y : array-like
        Coordinate y del robot.

    waypoints : ndarray, shape (N, 2)
        Coordinate dei waypoint.

    threshold : float
        Raggio della regione di visita [m].

    Returns
    -------
    visits : ndarray, shape (N,)
        Numero di visite per ciascun waypoint.
    """

    positions = np.column_stack(
        (x, y)
    )

    visits = np.zeros(
        len(waypoints),
        dtype=int,
    )

    for w, waypoint in enumerate(waypoints):

        distances = np.linalg.norm(
            positions - waypoint,
            axis=1,
        )

        inside = (
            distances <= threshold
        )

        if len(inside) == 0:
            continue

        # Se il primo campione è già dentro
        # il waypoint, conta una visita.
        visits[w] = int(inside[0])

        # Conta le transizioni:
        #
        # False -> True
        #
        # cioè ingresso nella regione.
        if len(inside) > 1:

            visits[w] += np.sum(
                (~inside[:-1])
                & inside[1:]
            )

    return visits


def make_figure(default_size):

    if args.ral:
        return plt.figure(
            figsize=FIGSIZE
        )

    return plt.figure(
        figsize=default_size
    )


def save_figure(
    output_folder,
    filename,
):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    if SAVE_PDF:

        pdf_name = (
            os.path.splitext(filename)[0]
            + ".pdf"
        )

        plt.savefig(
            os.path.join(
                output_folder,
                pdf_name,
            ),
            bbox_inches="tight",
        )


# ============================================================
# ELABORAZIONE DI UN SINGOLO ESPERIMENTO
# ============================================================

def process_experiment(
    experiment_folder,
    config_name,
):

    data_folder = experiment_folder

    output_folder = os.path.join(
        experiment_folder,
        "plots",
    )

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        f"Configurazione: {config_name}"
    )
    print(
        f"Esperimento: {experiment_folder}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Selezione waypoint della configurazione
    # --------------------------------------------------------

    if config_name not in WAYPOINTS:

        raise ValueError(
            f"Configurazione non riconosciuta: "
            f"{config_name}"
        )

    waypoints = WAYPOINTS[
        config_name
    ].copy()

    print()
    print("Waypoint utilizzati:")

    for i, wp in enumerate(waypoints):

        print(
            f"  WP{i+1}: "
            f"x={wp[0]:.2f}, "
            f"y={wp[1]:.2f}"
        )

    # --------------------------------------------------------
    # Calcolo automatico MAX_TIME dai file mission_times
    # --------------------------------------------------------

    mission_times = []

    for file in MISSION_TIME_FILES:

        path = os.path.join(
            data_folder,
            file,
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"File mission time non trovato: "
                f"{path}"
            )

        df_time = pd.read_csv(path)

        max_t = (
            df_time[
                "charging_completion_time"
            ].max()
        )

        mission_times.append(
            max_t
        )

    MAX_TIME = max(
        mission_times
    )

    print()
    print(
        f"MAX_TIME utilizzato: "
        f"{MAX_TIME:.2f} s"
    )

    # --------------------------------------------------------
    # Caricamento robot
    # --------------------------------------------------------

    robots = []

    for file in POSITION_FILES:

        path = os.path.join(
            data_folder,
            file,
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"File posizione non trovato: "
                f"{path}"
            )

        df = pd.read_csv(path)

        if MAX_TIME is not None:

            df = df[
                df["time"] <= MAX_TIME
            ]

        robots.append(
            df.reset_index(
                drop=True
            )
        )

    # ========================================================
    # Conteggio visite ai waypoint
    # ========================================================

    waypoint_visits = np.zeros(
        (
            len(robots),
            len(waypoints),
        ),
        dtype=int,
    )

    for r, robot in enumerate(robots):

        waypoint_visits[r] = (
            count_waypoint_visits(
                robot["x"].values,
                robot["y"].values,
                waypoints,
                WAYPOINT_RADIUS,
            )
        )

    # Totale visite effettuate
    # dai tre robot
    total_waypoint_visits = (
        waypoint_visits.sum(
            axis=0
        )
    )

    # ========================================================
    # Timeline comune
    # ========================================================

    common_time = (
        robots[0]["time"].values
    )

    if MAX_TIME is not None:

        common_time = common_time[
            common_time <= MAX_TIME
        ]

    interp_data = []

    for df in robots:

        interp = {

            "x": np.interp(
                common_time,
                df["time"],
                df["x"],
            ),

            "y": np.interp(
                common_time,
                df["time"],
                df["y"],
            ),

            "yaw": np.interp(
                common_time,
                df["time"],
                df["yaw"],
            ),
        }

        interp_data.append(
            interp
        )

    if len(common_time) < 2:

        raise RuntimeError(
            "Timeline insufficiente "
            f"per l'esperimento "
            f"{experiment_folder}"
        )

    dt = np.mean(
        np.diff(common_time)
    )

    total_time = (
        common_time[-1]
        - common_time[0]
    )

    # ========================================================
    # 1) Distanza dal centro dello sciame
    # ========================================================

    sum_distances = []

    distances = []

    for i in range(
        len(common_time)
    ):

        positions = np.array([

            [
                interp_data[0]["x"][i],
                interp_data[0]["y"][i],
            ],

            [
                interp_data[1]["x"][i],
                interp_data[1]["y"][i],
            ],

            [
                interp_data[2]["x"][i],
                interp_data[2]["y"][i],
            ],
        ])

        center = (
            positions.mean(
                axis=0
            )
        )

        dist = np.linalg.norm(
            positions - center,
            axis=1,
        )

        distances.append(
            dist
        )

        sum_distances.append(
            dist.sum()
        )

    # ========================================================
    # Swarm compactness
    # ========================================================

    make_figure(
        (8, 4)
    )

    plt.plot(
        common_time,
        sum_distances,
    )

    plt.grid(True)

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Sum distance from swarm center [m]"
    )

    plt.title(
        "Swarm compactness"
    )

    save_figure(
        output_folder,
        "swarm_compactness.png",
    )

    plt.close()

    # ========================================================
    # Distanza individuale dal centro dello sciame
    # ========================================================

    make_figure(
        (8, 4)
    )

    distances = np.array(
        distances
    )

    for j in range(3):

        plt.plot(
            common_time,
            distances[:, j],
            label=f"UAV {j+1}",
        )

    plt.grid(True)

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Distance from swarm center [m]"
    )

    plt.title(
        "Distance of each UAV from swarm center"
    )

    plt.legend()

    save_figure(
        output_folder,
        "individual_swarm_distance.png",
    )

    plt.close()

    # ========================================================
    # 2) Tempo entro 2.5 m da ciascun waypoint
    #
    # Ogni istante viene assegnato AL MASSIMO
    # a un waypoint, cioè al waypoint più vicino.
    # ========================================================

    n_wp = len(
        waypoints
    )

    percent_near_wp = np.zeros(
        (
            3,
            n_wp,
        )
    )

    for r, robot in enumerate(
        interp_data
    ):

        x = robot["x"]
        y = robot["y"]

        positions = np.column_stack(
            (x, y)
        )

        distances_wp = np.linalg.norm(
            positions[:, None, :]
            - waypoints[None, :, :],
            axis=2,
        )

        nearest_wp = np.argmin(
            distances_wp,
            axis=1,
        )

        nearest_distance = np.min(
            distances_wp,
            axis=1,
        )

        near_any_wp = (
            nearest_distance
            < WAYPOINT_RADIUS
        )

        for w in range(n_wp):

            near_this_wp = (
                near_any_wp
                & (
                    nearest_wp == w
                )
            )

            time_near = (
                near_this_wp.sum()
                * dt
            )

            percent_near_wp[
                r,
                w,
            ] = time_near

    # ========================================================
    # Waypoint distance plot
    # ========================================================

    make_figure(
        (8, 5)
    )

    robots_name = [
        "Robot1",
        "Robot2",
        "Robot3",
    ]

    bottom = np.zeros(3)

    # Manteniamo il comportamento
    # del file originale:
    #
    # il grafico mostra il tempo totale
    # vicino ai waypoint.

    for w in range(n_wp):

        bottom += (
            percent_near_wp[:, w]
        )

    plt.bar(
        robots_name,
        bottom,
    )

    plt.ylabel(
        "Time [s]"
    )

    plt.title(
        "Tempo trascorso entro 2.5 m dai waypoint"
    )

    save_figure(
        output_folder,
        "wpdist.png",
    )

    plt.close()

    # ========================================================
    # 3) Heading verso waypoint
    # ========================================================

    heading_percent = []

    threshold = np.deg2rad(
        HEADING_THRESHOLD_DEG
    )

    for robot in interp_data:

        x = robot["x"]
        y = robot["y"]
        yaw = robot["yaw"]

        good_heading = np.zeros(
            len(common_time),
            dtype=bool,
        )

        for i in range(
            len(common_time)
        ):

            pos = np.array([
                x[i],
                y[i],
            ])

            d = np.linalg.norm(
                waypoints - pos,
                axis=1,
            )

            nearest = np.argmin(
                d
            )

            vec = (
                waypoints[nearest]
                - pos
            )

            desired = np.arctan2(
                vec[1],
                vec[0],
            )

            error = abs(
                wrap_angle(
                    desired - yaw[i]
                )
            )

            if error < threshold:

                good_heading[i] = True

        heading_percent.append(
            100
            * good_heading.sum()
            * dt
            / total_time
        )

    # ========================================================
    # Heading plot
    # ========================================================

    make_figure(
        (6, 4)
    )

    plt.bar(
        [
            "Robot1",
            "Robot2",
            "Robot3",
        ],
        heading_percent,
    )

    plt.ylabel(
        "% mission"
    )

    plt.title(
        "Heading toward nearest waypoint (<45°)"
    )

    save_figure(
        output_folder,
        "heading.png",
    )

    plt.close()

    # ========================================================
    # Stampa risultati
    # ========================================================

    log_file = os.path.join(
        output_folder,
        "results1.log",
    )

    with open(
        log_file,
        "w",
    ) as f:

        text = (
            "========== RESULTS ==========\n\n"
        )

        print(
            text,
            end=""
        )

        f.write(
            text
        )

        for i in range(3):

            text = (

                f"Robot {i+1}\n"

                f"Time near waypoint : "
                f"{percent_near_wp[i].sum():.2f} s\n"

                f"Heading to waypoint: "
                f"{heading_percent[i]:.2f}%\n\n"
            )

            print(
                text,
                end=""
            )

            f.write(
                text
            )

        # ----------------------------------------------------
        # Waypoint visits
        # ----------------------------------------------------

        text = (
            "\n"
            "========== WAYPOINT VISITS ==========\n\n"
        )

        print(
            text,
            end=""
        )

        f.write(
            text
        )

        for i in range(3):

            text = (
                f"Robot {i+1}\n"
            )

            for w in range(
                len(waypoints)
            ):

                text += (

                    f"  WP{w+1}: "
                    f"{waypoint_visits[i, w]} "
                    f"visits\n"
                )

            text += "\n"

            print(
                text,
                end=""
            )

            f.write(
                text
            )

        text = (
            "Total visits per waypoint:\n"
        )

        for w in range(
            len(waypoints)
        ):

            text += (

                f"  WP{w+1}: "
                f"{total_waypoint_visits[w]} "
                f"visits\n"
            )

        text += "\n"

        print(
            text,
            end=""
        )

        f.write(
            text
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame({

        "Robot": [
            "Robot1",
            "Robot2",
            "Robot3",
        ],

        "NearWaypoint_%":
            percent_near_wp.sum(
                axis=1
            ),

        "Heading_%":
            heading_percent,
    })

    summary.to_csv(
        os.path.join(
            output_folder,
            "summary.csv",
        ),
        index=False,
    )

    # ========================================================
    # Dettaglio waypoint
    # ========================================================

    columns = [
        f"WP{i+1}"
        for i in range(
            len(waypoints)
        )
    ]

    detail = pd.DataFrame(
        percent_near_wp,
        columns=columns,
        index=[
            "Robot1",
            "Robot2",
            "Robot3",
        ],
    )

    detail.to_csv(
        os.path.join(
            output_folder,
            "waypoint_statistics.csv",
        )
    )

    # ========================================================
    # Conteggio visite ai waypoint
    # ========================================================

    visit_columns = [
        f"WP{i+1}"
        for i in range(
            len(waypoints)
        )
    ]

    visit_detail = pd.DataFrame(
        waypoint_visits,
        columns=visit_columns,
        index=[
            "Robot1",
            "Robot2",
            "Robot3",
        ],
    )

    # Aggiunge il totale dei tre robot
    visit_detail.loc["Total"] = (
        visit_detail.sum(
            axis=0
        )
    )

    visit_detail.to_csv(
        os.path.join(
            output_folder,
            "waypoint_visit_counts.csv",
        )
    )

    print(
        f"Output salvati in: "
        f"{output_folder}"
    )


# ============================================================
# SCOPERTA AUTOMATICA DEGLI ESPERIMENTI
# ============================================================

def find_experiments(
    config_folder
):

    experiments = []

    if not os.path.isdir(
        config_folder
    ):

        return experiments

    # Cerca sia exp_equal*
    # sia exp_reward*

    for name in sorted(
        os.listdir(
            config_folder
        )
    ):

        path = os.path.join(
            config_folder,
            name,
        )

        if not os.path.isdir(
            path
        ):

            continue

        if (
            name.startswith(
                "exp_equal"
            )
            or
            name.startswith(
                "exp_reward"
            )
        ):

            experiments.append(
                path
            )

    return experiments


# ============================================================
# MAIN
# ============================================================

all_experiments = []

for config_name in CONFIGURATIONS:

    config_folder = os.path.join(
        BASE_PATH,
        config_name,
    )

    experiments = find_experiments(
        config_folder
    )

    if not experiments:

        print(
            f"[WARNING] Nessun esperimento "
            f"trovato in {config_folder}"
        )

        continue

    print()
    print(
        "#" * 70
    )

    print(
        f"{config_name}: "
        f"trovati {len(experiments)} esperimenti"
    )

    print(
        "#" * 70
    )

    for experiment_folder in experiments:

        all_experiments.append(
            (
                config_name,
                experiment_folder,
            )
        )


# ============================================================
# ESECUZIONE DI TUTTE LE CONFIGURAZIONI
# ============================================================

failed = []

for (
    config_name,
    experiment_folder,
) in all_experiments:

    try:

        process_experiment(
            experiment_folder,
            config_name,
        )

    except Exception as e:

        print()

        print(
            f"[ERROR] "
            f"{config_name} / "
            f"{os.path.basename(experiment_folder)}: "
            f"{e}"
        )

        failed.append(
            (
                config_name,
                experiment_folder,
                str(e),
            )
        )


# ============================================================
# RISULTATI FINALI
# ============================================================

print()
print(
    "=" * 70
)

print(
    "ELABORAZIONE COMPLETATA"
)

print(
    "=" * 70
)

for config_name in CONFIGURATIONS:

    count = sum(
        1
        for cfg, _ in all_experiments
        if cfg == config_name
    )

    print(
        f"{config_name}: "
        f"{count} esperimenti trovati"
    )


# ============================================================
# ESPERIMENTI CON ERRORI
# ============================================================

if failed:

    print()
    print(
        "Esperimenti con errore:"
    )

    for (
        config_name,
        experiment_folder,
        error,
    ) in failed:

        print(
            f"  - "
            f"{config_name}/"
            f"{os.path.basename(experiment_folder)}: "
            f"{error}"
        )


# ============================================================
# SHOW
# ============================================================

if not args.noshow:

    plt.show()