from __future__ import annotations

import json

import experiments.summaries.armijo.summarize_wlra_reg_qr_grid as summarize_wlra_reg_qr_grid


def test_wlra_reg_qr_summary_markdown_highlights_best_config(tmp_path):
    raw_dir = tmp_path / "raws"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    records = [
        {
            "status": "success",
            "routine": "armijo_wlra_reg",
            "retraction": "qr",
            "config": {"beta": 0.5, "sigma": 0.25, "initial_step": 0.1, "max_backtracks": 200},
            "iteration_numbers": 5000,
            "lmbda": 0.01,
            "final_rmse": 0.4,
            "final_loss": 12.0,
            "final_data_loss": 11.0,
            "final_gradient_norm": 1.5,
            "final_x_norm": 2.5,
            "max_backtracks_hit": False,
            "final_backtracks": 3,
            "device": "cpu",
            "run_timestamp": "20260524_010101",
            "config_index": 1,
        },
        {
            "status": "success",
            "routine": "armijo_wlra_reg",
            "retraction": "qr",
            "config": {"beta": 0.9, "sigma": 0.5, "initial_step": 1.0, "max_backtracks": 200},
            "iteration_numbers": 5000,
            "lmbda": 0.01,
            "final_rmse": 0.2,
            "final_loss": 10.0,
            "final_data_loss": 9.0,
            "final_gradient_norm": 0.5,
            "final_x_norm": 3.5,
            "max_backtracks_hit": True,
            "final_backtracks": 7,
            "device": "cpu",
            "run_timestamp": "20260524_010101",
            "config_index": 2,
        },
    ]
    for index, record in enumerate(records):
        (raw_dir / f"armijo_wlra_reg_qr_test_{index}.json").write_text(json.dumps(record))

    output = summarize_wlra_reg_qr_grid.write_summary(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        timestamp="20260524_020202",
    )

    text = output.read_text()
    assert output.name == "regularized_qr_summary.md"
    assert "# Regularized WLRA QR Grid Search Summary" in text
    assert "- Device: cpu" in text
    assert "- Number of iterations: 5000" in text
    assert (
        "| beta | sigma | s | r | lambda | max-backtracks | retraction | final rmse | final loss | "
        "final data loss | final gradient norm | final x norm | max-backtracks hit | approx runtime | "
        "final backtracks |"
    ) in text
    assert "| **0.9** | **0.5** | **1** |  | **0.01** | **200** | **qr** | **0.2** |" in text
    assert "**9**" in text
    assert "**true**" in text
    assert "**7**" in text
