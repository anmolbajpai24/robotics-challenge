"""Read-only API over the eval_info.json files under day*/ and outputs/eval/.

Parses results that already exist on disk. Never writes into those dirs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Robotics Challenge Eval Results API")

# TODO: This is a bit restrictive, will update it from parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {"lerobot-env", ".git", "node_modules", "__pycache__"}

# eval_info.json records success/rewards but not which policy ran or the eval
# protocol, so those are set here. protocol stays "exploratory" unless a run is
# known to have used the frozen ruler (500 ep, batch 4, seed 1000, async off).
FROZEN_500 = {
    "day3/eval_final",                                   # from-scratch diffusion 90k, 39.8%
    "outputs/eval/2026-07-16/22-35-58_pusht_diffusion",  # pretrained diffusion baseline, 61.0%
    "day5/eval_100k",                                    # ACT-on-pusht 100k, 0.8% — cmd matches ruler (bash history)
}

# step -> eval run for the day3 diffusion-scratch checkpoint sweep.
# Only the 90k point (eval_final) ran the frozen ruler; the rest are 150-ep probes.
SKILL_CURVES = {
    "diffusion-scratch": [
        (10000, "day3/eval_010000"),
        (30000, "day3/eval_030000"),
        (50000, "day3/eval_050000"),
        (70000, "day3/eval_070000"),
        (90000, "day3/eval_final"),
    ],
}

# Runs whose launch command is in a committed .log, not just bash history.
# eval500.log covers 22-35-58; eval_sweep.log covers the four day3 checkpoints.
# See day9/provenance.md.
PROVENANCE_LOG = {
    "outputs/eval/2026-07-16/22-35-58_pusht_diffusion",
    "day3/eval_010000",
    "day3/eval_030000",
    "day3/eval_050000",
    "day3/eval_070000",
}


def assert_policy(run_id: str) -> str:
    if run_id.startswith("outputs/eval/"):
        return "pretrained-diffusion"
    if run_id.startswith("day3/"):
        return "diffusion-scratch"
    if run_id.startswith("day5/"):
        return "act-pusht"
    if run_id.startswith("day7/"):
        return "act-aloha-pretrained"
    return "unknown"


def assert_protocol(run_id: str) -> str:
    return "frozen-500" if run_id in FROZEN_500 else "exploratory"


def assert_provenance(run_id: str) -> str:
    return "log" if run_id in PROVENANCE_LOG else "bash_history"


class RunSummary(BaseModel):
    id: str
    policy: str
    env_type: str
    protocol: str
    provenance_source: str
    n_episodes: int
    success_rate: float
    avg_sum_reward: float
    avg_max_reward: float
    videos_path: str
    timestamp: str


class EpisodeResult(BaseModel):
    episode: int
    success: bool
    sum_reward: float
    max_reward: float


class RunDetail(RunSummary):
    eval_s: float
    eval_ep_s: float
    video_paths: list[str]
    episodes: list[EpisodeResult]

class SummaryCounts(BaseModel):
    runs: int
    policies: int
    envs: int

class Summary(BaseModel):
    counts: SummaryCounts
    frozen_table: list[RunSummary]

class RunMetrics(BaseModel):
    id: str
    policy: str
    protocol: str
    n_episodes: int
    success_rate: float
    avg_sum_reward: float
    avg_max_reward: float
    eval_s: float
    eval_ep_s: float
    # chart-ready parallel arrays (index-aligned)
    episode: list[int]
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]

class CurvePoint(BaseModel):
    step: int
    run_id: str
    success_rate: float
    n_episodes: int
    protocol: str


class SkillCurve(BaseModel):
    policy: str
    points: list[CurvePoint]


def find_eval_files() -> list[Path]:
    eval_files = []
    for eval_file in REPO_ROOT.rglob("eval_info.json"):
        if any(part in EXCLUDE_DIRS for part in eval_file.parts):
            continue
        eval_files.append(eval_file)
    return eval_files


def run_id_for(path: Path) -> str:
    return path.parent.relative_to(REPO_ROOT).as_posix()


def env_type_of(data: dict) -> str:
    per_task = data.get("per_task") or []
    if per_task:
        return per_task[0].get("task_group", "unknown")
    per_group = data.get("per_group") or {}
    return next(iter(per_group), "unknown")


def summarize(path: Path) -> RunSummary:
    data = json.loads(path.read_text())
    overall = data["overall"]
    run_id = run_id_for(path)
    modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return RunSummary(
        id=run_id,
        policy=assert_policy(run_id),
        env_type=env_type_of(data),
        protocol=assert_protocol(run_id),
        provenance_source=assert_provenance(run_id),
        n_episodes=overall["n_episodes"],
        success_rate=overall["pc_success"],
        avg_sum_reward=overall["avg_sum_reward"],
        avg_max_reward=overall["avg_max_reward"],
        videos_path=f"{run_id}/videos",
        timestamp=modified_time,
    )


def resolve_run_path(run_id: str) -> Path:
    if ".." in run_id.split("/") or run_id.startswith("/"):
        raise HTTPException(status_code=404, detail=f"No run with id '{run_id}'")
    path = REPO_ROOT / run_id / "eval_info.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No run with id '{run_id}'")
    return path


def load_detail(run_id: str) -> RunDetail:
    path = resolve_run_path(run_id)
    data = json.loads(path.read_text())
    overall = data["overall"]
    summary = summarize(path)

    metrics = (data.get("per_task") or [{}])[0].get("metrics", {})
    sum_rewards = metrics.get("sum_rewards", [])
    max_rewards = metrics.get("max_rewards", [])
    successes = metrics.get("successes", [])
    episodes = [
        EpisodeResult(
            episode=index,
            success=bool(is_success),
            sum_reward=float(sum_reward),
            max_reward=float(max_reward),
        )
        for index, (is_success, sum_reward, max_reward) in enumerate(
            zip(successes, sum_rewards, max_rewards)
        )
    ]

    return RunDetail(
        **summary.model_dump(),
        eval_s=overall.get("eval_s", 0.0),
        eval_ep_s=overall.get("eval_ep_s", 0.0),
        video_paths=overall.get("video_paths", []),
        episodes=episodes,
    )


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/summary", response_model=Summary)
def summary():
    runs = [summarize(eval_file) for eval_file in find_eval_files()]
    frozen_table = [run for run in runs if run.protocol == "frozen-500"]
    frozen_table.sort(key=lambda run: run.success_rate, reverse=True)
    return Summary(
        counts=SummaryCounts(
            runs=len(runs),
            policies=len(set(run.policy for run in runs)),
            envs=len(set(run.env_type for run in runs)),
        ),
        frozen_table=frozen_table,
    )


@app.get("/runs", response_model=list[RunSummary])
def list_runs():
    runs = [summarize(eval_file) for eval_file in find_eval_files()]
    runs.sort(key=lambda run: run.timestamp, reverse=True)
    return runs

@app.get("/runs/{run_id:path}/metrics", response_model=RunMetrics)
def get_run_metrics(run_id: str):
    eval_path = resolve_run_path(run_id)
    eval_data = json.loads(eval_path.read_text())
    overall = eval_data["overall"]
    summary = summarize(eval_path)

    episode_metrics = (eval_data.get("per_task") or [{}])[0].get("metrics", {})
    sum_rewards = episode_metrics.get("sum_rewards", [])
    max_rewards = episode_metrics.get("max_rewards", [])
    successes = episode_metrics.get("successes", [])

    return RunMetrics(
        id=summary.id,
        policy=summary.policy,
        protocol=summary.protocol,
        n_episodes=summary.n_episodes,
        success_rate=summary.success_rate,
        avg_sum_reward=summary.avg_sum_reward,
        avg_max_reward=summary.avg_max_reward,
        eval_s=overall.get("eval_s", 0.0),
        eval_ep_s=overall.get("eval_ep_s", 0.0),
        episode=list(range(len(successes))),
        sum_rewards=[float(sum_reward) for sum_reward in sum_rewards],
        max_rewards=[float(max_reward) for max_reward in max_rewards],
        successes=[bool(is_success) for is_success in successes],
    )

@app.get("/runs/{run_id:path}", response_model=RunDetail)
def get_run(run_id: str):
    return load_detail(run_id)

@app.get("/curves/skill", response_model=SkillCurve)
def get_skill_curve(policy: str = "diffusion-scratch"):
    checkpoints = SKILL_CURVES.get(policy)
    if checkpoints is None:
        raise HTTPException(status_code=404, detail=f"No skill curve for policy '{policy}'")

    points = []
    for step, run_id in checkpoints:
        eval_path = resolve_run_path(run_id)
        summary = summarize(eval_path)
        points.append(
            CurvePoint(
                step=step,
                run_id=run_id,
                success_rate=summary.success_rate,
                n_episodes=summary.n_episodes,
                protocol=summary.protocol,
            )
        )
    return SkillCurve(policy=policy, points=points)
