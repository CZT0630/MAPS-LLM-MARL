from LLM4RL.experiments.ei.formal_matrix import (
    build_evaluation_jobs,
    build_matrix_plan,
    build_training_jobs,
    load_matrix_config,
)


def test_formal_matrix_training_plan_matches_c7_budget():
    matrix = load_matrix_config("configs/ei/formal_matrix.yaml")
    jobs = build_training_jobs(matrix)

    assert len(jobs) == 65
    assert sum(job.experiment == "e1" for job in jobs) == 20
    assert sum(job.experiment == "e3" for job in jobs) == 45
    assert {job.seed for job in jobs} == {42, 43, 44, 45, 46}
    assert {job.algorithm for job in jobs if job.experiment == "e1"} == {
        "maddpg",
        "mappo",
        "maps_no_annealing",
        "maps",
    }
    assert {job.algorithm for job in jobs if job.experiment == "e3"} == {
        "maddpg",
        "mappo",
        "maps",
    }
    assert "maps_no_annealing" not in {
        job.algorithm for job in jobs if job.experiment == "e3"
    }


def test_formal_matrix_evaluation_reuses_expected_checkpoints():
    matrix = load_matrix_config("configs/ei/formal_matrix.yaml")
    jobs = build_evaluation_jobs(matrix)

    assert len(jobs) == 118
    e2_learning = [
        job
        for job in jobs
        if job.experiment == "e2" and job.algorithm in {"maddpg", "mappo", "maps"}
    ]
    assert e2_learning
    assert all(job.checkpoint_ref.startswith("train:e1:s1_u10_medium") for job in e2_learning)

    e3_learning = [
        job
        for job in jobs
        if job.experiment == "e3" and job.algorithm in {"maddpg", "mappo", "maps"}
    ]
    assert e3_learning
    assert all(
        job.checkpoint_ref == f"train:e3:{job.scenario}:{job.algorithm}:seed{job.seed}"
        for job in e3_learning
    )

    e4_learning = [
        job
        for job in jobs
        if job.experiment == "e4" and job.algorithm in {"maddpg", "mappo", "maps"}
    ]
    assert e4_learning
    assert all(job.checkpoint_ref.startswith("train:e1:s1_u10_medium") for job in e4_learning)
    assert {"greedy_min_cost", "llm_only"}.issubset(
        {job.algorithm for job in jobs if job.experiment == "e2"}
    )
    assert "llm_only" not in {job.algorithm for job in jobs if job.experiment == "e3"}


def test_formal_matrix_plan_is_serializable_and_counts_jobs():
    matrix = load_matrix_config("configs/ei/formal_matrix.yaml")
    plan = build_matrix_plan(matrix)

    assert plan["schema_version"] == "ei-formal-matrix-plan-v1"
    assert plan["training_job_count"] == 65
    assert plan["evaluation_job_count"] == 118
    assert plan["training_jobs"][0]["kind"] == "train"
    assert plan["evaluation_jobs"][0]["kind"] == "eval"
