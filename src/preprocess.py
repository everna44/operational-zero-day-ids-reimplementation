from pathlib import Path

import pandas as pd


# Project paths
RAW_DIR = Path("data/raw/CICIDS2017_improved")
PROCESSED_DIR = Path("data/processed")

NON_FEATURE_COLUMNS = {
    "id",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
    "Attempted Category",
    "traffic_category",
}


def get_flow_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return numeric traffic-statistic columns used for
    host-session feature construction.
    """
    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    return feature_columns


def map_traffic_category(label: str) -> str:
    """
    Map raw CICIDS2017 labels to the three categories used in this project.

    BENIGN:
        Normal traffic.

    OOD:
        Botnet family held out as unseen/zero-day traffic.

    SEEN:
        All remaining attack families.
    """
    label = str(label).strip()

    if label == "BENIGN":
        return "BENIGN"

    if label.startswith("Botnet"):
        return "OOD"

    return "SEEN"


def load_flow_file(csv_path: Path) -> pd.DataFrame:
    """
    Load one raw CICIDS2017 CSV file and add the project-level traffic category.
    """
    df = pd.read_csv(csv_path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
    )

    df["traffic_category"] = df["Label"].map(
        map_traffic_category
    )

    return df


def build_endpoint_table(
    df: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """
    Convert flow records into endpoint-level records.

    Each flow is represented twice:
    - once from the source host perspective
    - once from the destination host perspective

    The resulting table is used as the basis for host-session
    aggregation and communication-edge construction.
    """
    if window_size not in {"1min", "5min"}:
        raise ValueError(
            "window_size must be either '1min' or '5min'"
        )

    window = df["Timestamp"].dt.floor(window_size)

    src = pd.DataFrame(
        {
            "host": df["Src IP"],
            "peer": df["Dst IP"],
            "window": window,
            "direction": "out",
            "traffic_category": df["traffic_category"],
        }
    )

    dst = pd.DataFrame(
        {
            "host": df["Dst IP"],
            "peer": df["Src IP"],
            "window": window,
            "direction": "in",
            "traffic_category": df["traffic_category"],
        }
    )

    endpoints = pd.concat(
        [src, dst],
        ignore_index=True,
    )

    return endpoints


def build_host_sessions(
    endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one host-session row for each unique (host, window).

    Label priority:
        OOD > SEEN > BENIGN
    """
    priority = {
        "BENIGN": 0,
        "SEEN": 1,
        "OOD": 2,
    }

    reverse_priority = {
        0: "BENIGN",
        1: "SEEN",
        2: "OOD",
    }

    sessions = endpoints[
        ["host", "window", "traffic_category"]
    ].copy()

    sessions["label_priority"] = sessions[
        "traffic_category"
    ].map(priority)

    sessions = (
        sessions
        .groupby(
            ["host", "window"],
            as_index=False,
        )["label_priority"]
        .max()
    )

    sessions["session_category"] = sessions[
        "label_priority"
    ].map(reverse_priority)

    sessions = sessions.drop(
        columns=["label_priority"]
    )

    return sessions


def aggregate_directional_features(
    df: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """
    Aggregate flow-level numeric features into directional
    host-session features.

    For each host and time window:
    - out_* features summarize flows where the host is Src IP.
    - in_* features summarize flows where the host is Dst IP.

    The current reimplementation uses the mean of each
    flow-level numeric feature within a host-session.
    """
    if window_size not in {"1min", "5min"}:
        raise ValueError(
            "window_size must be either '1min' or '5min'"
        )

    feature_columns = get_flow_feature_columns(df)

    numeric = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    working = pd.concat(
        [
            df[
                [
                    "Src IP",
                    "Dst IP",
                    "Timestamp",
                ]
            ].reset_index(drop=True),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )

    working["window"] = working[
        "Timestamp"
    ].dt.floor(window_size)

    outgoing = (
        working
        .groupby(
            ["Src IP", "window"],
            as_index=False,
        )[feature_columns]
        .mean()
        .rename(
            columns={"Src IP": "host"}
        )
    )

    outgoing = outgoing.rename(
        columns={
            column: f"out_{column}"
            for column in feature_columns
        }
    )

    incoming = (
        working
        .groupby(
            ["Dst IP", "window"],
            as_index=False,
        )[feature_columns]
        .mean()
        .rename(
            columns={"Dst IP": "host"}
        )
    )

    incoming = incoming.rename(
        columns={
            column: f"in_{column}"
            for column in feature_columns
        }
    )

    features = outgoing.merge(
        incoming,
        on=["host", "window"],
        how="outer",
    )

    return features


def fill_missing_directional_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing directional features with zero.

    Missing values occur when a host-session has traffic
    only in one direction:
    - no outgoing flows -> out_* features are missing
    - no incoming flows -> in_* features are missing
    """
    features = features.copy()

    feature_columns = [
        column
        for column in features.columns
        if column.startswith("out_")
        or column.startswith("in_")
    ]

    features[feature_columns] = features[
        feature_columns
    ].fillna(0.0)

    return features


def build_host_session_table(
    df: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """
    Build the final host-session node table containing:
    - host identifier
    - time window
    - directional traffic features
    - BENIGN / SEEN / OOD session label
    """
    endpoints = build_endpoint_table(
        df,
        window_size,
    )

    sessions = build_host_sessions(
        endpoints,
    )

    features = aggregate_directional_features(
        df,
        window_size,
    )

    features = fill_missing_directional_features(
        features,
    )

    node_table = features.merge(
        sessions,
        on=["host", "window"],
        how="left",
        validate="one_to_one",
    )

    if node_table["session_category"].isna().any():
        raise RuntimeError(
            "Some host-sessions are missing labels."
        )

    return node_table


def process_and_save_file(
    csv_path: Path,
    window_size: str,
) -> Path:
    """
    Process one raw CICIDS2017 CSV file and save the resulting
    host-session node table.

    Example output:
        data/processed/friday_1m_nodes.csv
        data/processed/friday_5m_nodes.csv
    """
    df = load_flow_file(csv_path)

    node_table = build_host_session_table(
        df,
        window_size,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_name = {
        "1min": "1m",
        "5min": "5m",
    }[window_size]

    output_path = (
        PROCESSED_DIR
        / f"{csv_path.stem}_{window_name}_nodes.csv"
    )

    node_table.to_csv(
        output_path,
        index=False,
    )

    return output_path


def process_all_files() -> None:
    """
    Process all CICIDS2017 daily CSV files for both
    1-minute and 5-minute host-session windows.
    """
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    for csv_path in csv_files:
        for window_size in ["1min", "5min"]:
            print(
                f"Processing {csv_path.name} "
                f"with window={window_size}..."
            )

            output_path = process_and_save_file(
                csv_path,
                window_size,
            )

            print(f"Saved: {output_path}")


def combine_node_files(
    window_name: str,
) -> Path:
    """
    Combine daily host-session node files into one dataset
    for a given window size and assign deterministic node IDs.

    Example outputs:
        data/processed/nodes_1m.csv
        data/processed/nodes_5m.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    node_files = sorted(
        PROCESSED_DIR.glob(
            f"*_{window_name}_nodes.csv"
        )
    )

    if not node_files:
        raise FileNotFoundError(
            f"No {window_name} node files found."
        )

    frames = []

    for path in node_files:
        df = pd.read_csv(path)

        source_day = path.name.replace(
            f"_{window_name}_nodes.csv",
            "",
        )

        df["source_day"] = source_day

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["window"] = pd.to_datetime(
        combined["window"],
        errors="raise",
    )

    combined = (
        combined
        .sort_values(
            ["window", "host"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if combined.duplicated(
        ["host", "window"]
    ).any():
        raise RuntimeError(
            "Duplicate host-window pairs found "
            "after combining node files."
        )

    combined.insert(
        0,
        "node_id",
        range(len(combined)),
    )

    output_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return output_path


def build_communication_edges(
    df: pd.DataFrame,
    node_table: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """
    Build undirected communication edges between host-session nodes.

    Each raw flow connects the source and destination host-session
    within the same aggregation window.

    Repeated communications between the same node pair are merged,
    and interaction_count records the number of raw flows.
    """
    if window_size not in {"1min", "5min"}:
        raise ValueError(
            "window_size must be either '1min' or '5min'"
        )

    nodes = node_table[
        ["node_id", "host", "window"]
    ].copy()

    nodes["window"] = pd.to_datetime(
        nodes["window"],
        errors="raise",
    )

    flows = pd.DataFrame({
        "src_host": df["Src IP"],
        "dst_host": df["Dst IP"],
        "window": df["Timestamp"].dt.floor(
            window_size
        ),
    })

    src_lookup = nodes.rename(
        columns={
            "host": "src_host",
            "node_id": "src_node_id",
        }
    )

    dst_lookup = nodes.rename(
        columns={
            "host": "dst_host",
            "node_id": "dst_node_id",
        }
    )

    flows = flows.merge(
        src_lookup[
            ["src_host", "window", "src_node_id"]
        ],
        on=["src_host", "window"],
        how="left",
        validate="many_to_one",
    )

    flows = flows.merge(
        dst_lookup[
            ["dst_host", "window", "dst_node_id"]
        ],
        on=["dst_host", "window"],
        how="left",
        validate="many_to_one",
    )

    if flows[
        ["src_node_id", "dst_node_id"]
    ].isna().any().any():
        raise RuntimeError(
            "Some flow endpoints could not be mapped "
            "to host-session node IDs."
        )

    flows["src_node_id"] = (
        flows["src_node_id"].astype("int64")
    )
    flows["dst_node_id"] = (
        flows["dst_node_id"].astype("int64")
    )

    # Remove raw self-loops.
    flows = flows[
        flows["src_node_id"]
        != flows["dst_node_id"]
    ].copy()

    # Canonical ordering makes the graph undirected.
    flows["edge_src"] = flows[
        ["src_node_id", "dst_node_id"]
    ].min(axis=1)

    flows["edge_dst"] = flows[
        ["src_node_id", "dst_node_id"]
    ].max(axis=1)

    edges = (
        flows.groupby(
            ["edge_src", "edge_dst"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "edge_src": "src_node_id",
                "edge_dst": "dst_node_id",
                "size": "interaction_count",
            }
        )
    )

    return edges


def process_and_save_edges(
    csv_path: Path,
    window_size: str,
) -> Path:
    """
    Build and save communication edges for one raw daily CSV file.

    Example outputs:
        data/processed/friday_1m_edges.csv
        data/processed/friday_5m_edges.csv
    """
    if window_size not in {"1min", "5min"}:
        raise ValueError(
            "window_size must be either '1min' or '5min'"
        )

    window_name = {
        "1min": "1m",
        "5min": "5m",
    }[window_size]

    raw = load_flow_file(
        csv_path,
    )

    node_path = (
        PROCESSED_DIR
        / f"nodes_{window_name}.csv"
    )

    nodes = pd.read_csv(
        node_path,
    )

    nodes = nodes[
        nodes["source_day"] == csv_path.stem
    ].copy()

    edges = build_communication_edges(
        raw,
        nodes,
        window_size,
    )

    output_path = (
        PROCESSED_DIR
        / f"{csv_path.stem}_{window_name}_edges.csv"
    )

    edges.to_csv(
        output_path,
        index=False,
    )

    return output_path


def process_all_edges() -> None:
    """
    Build and save communication-edge files for all daily CSV files
    using both 1-minute and 5-minute aggregation windows.
    """
    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    for csv_path in csv_files:
        for window_size in ["1min", "5min"]:
            print(
                f"Processing edges for {csv_path.name} "
                f"with window={window_size}..."
            )

            output_path = process_and_save_edges(
                csv_path,
                window_size,
            )

            print(f"Saved: {output_path}")


def combine_edge_files(
    window_name: str,
) -> Path:
    """
    Combine daily communication-edge files into one graph
    for a given window size.

    Example outputs:
        data/processed/edges_1m.csv
        data/processed/edges_5m.csv
    """
    if window_name not in {"1m", "5m"}:
        raise ValueError(
            "window_name must be either '1m' or '5m'"
        )

    edge_files = sorted(
        PROCESSED_DIR.glob(
            f"*_{window_name}_edges.csv"
        )
    )

    if not edge_files:
        raise FileNotFoundError(
            f"No {window_name} edge files found."
        )

    frames = []

    for path in edge_files:
        edges = pd.read_csv(path)
        frames.append(edges)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.duplicated(
        ["src_node_id", "dst_node_id"]
    ).any():
        raise RuntimeError(
            "Duplicate edges found after combining."
        )

    if (
        combined["src_node_id"]
        == combined["dst_node_id"]
    ).any():
        raise RuntimeError(
            "Self-loops found after combining."
        )

    output_path = (
        PROCESSED_DIR
        / f"edges_{window_name}.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return output_path