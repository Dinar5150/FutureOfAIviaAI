import os
import pickle

from model import run_link_prediction_loaded


if __name__ == "__main__":
    # Parameter grids match M4/M5 style.
    delta_list = [1, 3, 5]
    cutoff_list = [0, 5, 25]
    min_edges_list = [1, 3]

    header = (
        "- Prediction from year (2021 - delta, 2021), with delta = [1, 3, 5]\n"
        "- Minimal vertex degree: cutoff = [0, 5, 25]\n"
        "- Prediction from unconnected to edge_weight = [1, 3] edges\n\n"
        "(delta, cutoff, edge_weight):\n"
    )

    with open("AUC Summary M9.txt", "w", encoding="utf-8") as log_file:
        log_file.write(header)

    for current_min_edges in min_edges_list:
        for curr_vertex_degree_cutoff in cutoff_list:
            for current_delta in delta_list:
                data_source = (
                    "SemanticGraph_delta_"
                    + str(current_delta)
                    + "_cutoff_"
                    + str(curr_vertex_degree_cutoff)
                    + "_minedge_"
                    + str(current_min_edges)
                    + ".pkl"
                )

                if not os.path.isfile(data_source):
                    print("File", data_source, "does not exist. Proceed to next parameter setting.")
                    continue

                with open(data_source, "rb") as pkl_file:
                    (
                        full_dynamic_graph_sparse,
                        unconnected_vertex_pairs,
                        unconnected_vertex_pairs_solution,
                        year_start,
                        years_delta,
                        vertex_degree_cutoff,
                        min_edges,
                    ) = pickle.load(pkl_file)

                res = run_link_prediction_loaded(
                    full_dynamic_graph_sparse=full_dynamic_graph_sparse,
                    unconnected_vertex_pairs=unconnected_vertex_pairs,
                    unconnected_vertex_pairs_solution=unconnected_vertex_pairs_solution,
                    year_start=year_start,
                    config={"max_pairs": 200000} # Cap for speed
                )

                auc_full = res.get("auc_full")
                auc_val = res.get("auc_val")
                print(
                    f"M9 AUC (full/val) for delta={current_delta}, cutoff={curr_vertex_degree_cutoff}, "
                    f"min_edges={current_min_edges}: {auc_full}"
                    + (f" / {auc_val}" if auc_val is not None else "")
                )

                with open("AUC Summary M9.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"- ({current_delta}, {curr_vertex_degree_cutoff}, {current_min_edges}): "
                        f"AUC_full = {auc_full}, AUC_val = {auc_val}\n"
                    )
