"""Read-only API over the eval_info.json files under day*/ and outputs/eval/.

Parses results that already exist on disk. Never writes into those dirs.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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


# Curated rollout clips for the Episodes player, one per storyline. The
# allowlist doubles as the path check in /videos/file — nothing outside it
# is ever served.
VIDEO_CLIPS = [
    ("pretrained-diffusion · PushT ep0, success (61.0% frozen-500)",
     "outputs/eval/2026-07-16/22-35-58_pusht_diffusion/videos/pusht_0/eval_episode_0.mp4"),
    ("diffusion-scratch 90k · PushT ep0, success (39.8% frozen-500)",
     "day3/eval_final/videos/pusht_0/eval_episode_0.mp4"),
    ("act-pusht · PushT ep0, a rare success (0.8% frozen-500)",
     "day5/eval_100k/videos/pusht_0/eval_episode_0.mp4"),
    ("act-aloha-pretrained · ALOHA transfer cube, best episode (7/10)",
     "day7/aloha_transfer_best_ep3.mp4"),
    ("bc-mlp-sighted · xarm lift ep0 (99.0% frozen-500)",
     "day16/rollout_sighted_ep0.mp4"),
    ("bc-mlp-blind-control · xarm lift ep0 (4.2% frozen-500)",
     "day16/rollout_blind_ep0.mp4"),
    ("bc-mlp-push · xarm push ep0 (98.8% frozen-500)",
     "day21/rollout_push_ep0.mp4"),
]


# The xarm lift/push BC runs (day14, day16, day21) were scored with a
# hand-written eval loop (day14/eval_bc.py and its day16/day21 copies), not
# lerobot_eval.py, so there's no eval_info.json for them, only a plain-text
# log. These four logs are all on the frozen-500 protocol (day14/EVAL_PROTOCOL.md):
# 500 episodes, reset(seed=1000+i), 300-step cap. Read-only: this only reads
# the committed .txt files below, never writes to them.
XARM_TXT_RUNS = [
    {
        "id": "day14/eval_500_seed1000",
        "policy": "bc-mlp-blind",
        "txt_path": "day14/eval_500_seed1000.txt",
        "video_paths": [
            "day14/rollout_ep0.mp4", "day14/rollout_ep1.mp4", "day14/rollout_ep2.mp4",
        ],
    },
    {
        "id": "day16/eval_500_blind",
        "policy": "bc-mlp-blind-control",
        "txt_path": "day16/eval_500_blind.txt",
        "video_paths": [
            "day16/rollout_blind_ep0.mp4", "day16/rollout_blind_ep1.mp4", "day16/rollout_blind_ep2.mp4",
        ],
    },
    {
        "id": "day16/eval_500_sighted",
        "policy": "bc-mlp-sighted",
        "txt_path": "day16/eval_500_sighted.txt",
        "video_paths": [
            "day16/rollout_sighted_ep0.mp4", "day16/rollout_sighted_ep1.mp4", "day16/rollout_sighted_ep2.mp4",
        ],
    },
    {
        "id": "day21/eval_500_push",
        "policy": "bc-mlp-push",
        "txt_path": "day21/eval_500_push.txt",
        "video_paths": [
            "day21/rollout_push_ep0.mp4", "day21/rollout_push_ep1.mp4", "day21/rollout_push_ep2.mp4",
        ],
    },
]
XARM_RUNS_BY_ID = {entry["id"]: entry for entry in XARM_TXT_RUNS}

# "RESULT: 0/500 = 0.0% success | seed base 1000 | 85.3s (0.17s/episode)" (day14)
# "RESULT [push]: 494/500 = 98.8% success | seed base 1000 | 31.4s (0.06s/episode)" (day16/day21)
XARM_RESULT_LINE = re.compile(
    r"RESULT(?:\s*\[\w+\])?:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%\s*success\s*\|"
    r"\s*seed base (\d+)\s*\|\s*([\d.]+)s\s*\(([\d.]+)s/episode\)"
)
# "episode   1/500  seed 1000  steps 300  success False  running rate 0.0%"
XARM_EPISODE_LINE = re.compile(
    r"^episode\s+\d+/\d+\s+seed\s+\d+\s+steps\s+\d+\s+success\s+(True|False)\s+running rate"
)


def parse_xarm_txt(txt_path: str) -> dict:
    """Read-only parse of one day14/16/21 eval_bc.py log. Never writes to it."""
    text = (REPO_ROOT / txt_path).read_text()
    result_match = XARM_RESULT_LINE.search(text)
    if result_match is None:
        raise HTTPException(status_code=500, detail=f"no RESULT line in {txt_path}")
    n_success, n_episodes, success_rate, seed_base, eval_s, eval_ep_s = result_match.groups()
    successes = [
        line_match.group(1) == "True"
        for line_match in (XARM_EPISODE_LINE.match(line) for line in text.splitlines())
        if line_match
    ]
    return {
        "n_episodes": int(n_episodes),
        "n_success": int(n_success),
        "success_rate": float(success_rate),
        "seed_base": int(seed_base),
        "eval_s": float(eval_s),
        "eval_ep_s": float(eval_ep_s),
        "successes": successes,
    }


def xarm_run_summary(entry: dict) -> RunSummary:
    path = REPO_ROOT / entry["txt_path"]
    parsed = parse_xarm_txt(entry["txt_path"])
    modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    # day14/16/21 success is binary (cube height or cube-to-goal distance past
    # a threshold), there is no graded reward like PushT's. To keep the
    # RunSummary schema uniform, avg_sum_reward/avg_max_reward here are the
    # same binary success fraction already in success_rate, not a new number.
    reward_proxy = parsed["n_success"] / parsed["n_episodes"]
    return RunSummary(
        id=entry["id"],
        policy=entry["policy"],
        env_type="xarm",
        protocol="frozen-500",
        provenance_source="log",
        n_episodes=parsed["n_episodes"],
        success_rate=parsed["success_rate"],
        avg_sum_reward=reward_proxy,
        avg_max_reward=reward_proxy,
        videos_path=entry["txt_path"].rsplit("/", 1)[0],
        timestamp=modified_time,
    )


def xarm_run_detail(entry: dict) -> RunDetail:
    summary = xarm_run_summary(entry)
    parsed = parse_xarm_txt(entry["txt_path"])
    return RunDetail(
        **summary.model_dump(),
        eval_s=parsed["eval_s"],
        eval_ep_s=parsed["eval_ep_s"],
        video_paths=[p for p in entry["video_paths"] if (REPO_ROOT / p).is_file()],
        episodes=[
            EpisodeResult(
                episode=index, success=is_success,
                sum_reward=1.0 if is_success else 0.0,
                max_reward=1.0 if is_success else 0.0,
            )
            for index, is_success in enumerate(parsed["successes"])
        ],
    )


def xarm_run_metrics(entry: dict) -> RunMetrics:
    summary = xarm_run_summary(entry)
    parsed = parse_xarm_txt(entry["txt_path"])
    successes = parsed["successes"]
    return RunMetrics(
        id=summary.id,
        policy=summary.policy,
        protocol=summary.protocol,
        n_episodes=summary.n_episodes,
        success_rate=summary.success_rate,
        avg_sum_reward=summary.avg_sum_reward,
        avg_max_reward=summary.avg_max_reward,
        eval_s=parsed["eval_s"],
        eval_ep_s=parsed["eval_ep_s"],
        episode=list(range(len(successes))),
        sum_rewards=[1.0 if is_success else 0.0 for is_success in successes],
        max_rewards=[1.0 if is_success else 0.0 for is_success in successes],
        successes=successes,
    )


def find_xarm_runs() -> list[dict]:
    return [entry for entry in XARM_TXT_RUNS if (REPO_ROOT / entry["txt_path"]).is_file()]


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


class VideoClip(BaseModel):
    id: str
    title: str
    url: str


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
    runs += [xarm_run_summary(entry) for entry in find_xarm_runs()]
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
    runs += [xarm_run_summary(entry) for entry in find_xarm_runs()]
    runs.sort(key=lambda run: run.timestamp, reverse=True)
    return runs

@app.get("/runs/{run_id:path}/metrics", response_model=RunMetrics)
def get_run_metrics(run_id: str):
    if run_id in XARM_RUNS_BY_ID:
        return xarm_run_metrics(XARM_RUNS_BY_ID[run_id])
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
    if run_id in XARM_RUNS_BY_ID:
        return xarm_run_detail(XARM_RUNS_BY_ID[run_id])
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


@app.get("/videos", response_model=list[VideoClip])
def list_videos():
    return [
        VideoClip(id=rel_path, title=title, url=f"/videos/file/{rel_path}")
        for title, rel_path in VIDEO_CLIPS
        if (REPO_ROOT / rel_path).is_file()
    ]


@app.get("/videos/file/{rel_path:path}")
def get_video_file(rel_path: str):
    if rel_path not in {path for _, path in VIDEO_CLIPS}:
        raise HTTPException(status_code=404, detail=f"No video at '{rel_path}'")
    video_path = REPO_ROOT / rel_path
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"No video at '{rel_path}'")
    return FileResponse(video_path, media_type="video/mp4")


# The built frontend (dashboard/static, from `npm run build` in dashboard/frontend).
# Mounted last so every API route above wins; html=True serves index.html at /.
# Conditional so the bare API still runs before the frontend is built.
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
