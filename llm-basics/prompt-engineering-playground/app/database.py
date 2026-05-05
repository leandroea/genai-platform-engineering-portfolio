"""
Database module for Prompt Engineering Playground.
Handles SQLite operations for storing and retrieving experiments.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.models import Experiment, ComparisonResult, LLMParameters


def get_db_path() -> Path:
    """Get the path to the SQLite database."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "experiments.db"


def init_db():
    """Initialize the database with required tables."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            parameters TEXT NOT NULL,
            response TEXT,
            rating INTEGER,
            feedback TEXT,
            experiment_type TEXT NOT NULL,
            name TEXT
        )
    """)

    # Create comparison_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comparison_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            prompt_index INTEGER NOT NULL,
            response TEXT NOT NULL,
            rating INTEGER,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def save_experiment(
    prompt: str,
    parameters: LLMParameters,
    response: str,
    experiment_type: str = "single",
    name: str = None,
    rating: int = None,
    feedback: str = None
) -> int:
    """Save an experiment to the database."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    params_json = json.dumps(parameters.model_dump())
    created_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO experiments (created_at, prompt, parameters, response, rating, feedback, experiment_type, name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (created_at, prompt, params_json, response, rating, feedback, experiment_type, name))

    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return experiment_id


def save_comparison_result(
    experiment_id: int,
    prompt_index: int,
    response: str,
    rating: int = None
) -> int:
    """Save a comparison result to the database."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO comparison_results (experiment_id, prompt_index, response, rating)
        VALUES (?, ?, ?, ?)
    """, (experiment_id, prompt_index, response, rating))

    result_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return result_id


def get_all_experiments(
    experiment_type: str = None,
    min_rating: int = None,
    search_text: str = None,
    limit: int = 50
) -> list[Experiment]:
    """Get all experiments with optional filtering."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM experiments WHERE 1=1"
    params = []

    if experiment_type:
        query += " AND experiment_type = ?"
        params.append(experiment_type)

    if min_rating:
        query += " AND rating >= ?"
        params.append(min_rating)

    if search_text:
        query += " AND (prompt LIKE ? OR name LIKE ?)"
        params.append(f"%{search_text}%")
        params.append(f"%{search_text}%")

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    experiments = []
    for row in rows:
        experiment = Experiment(
            id=row["id"],
            created_at=row["created_at"],
            prompt=row["prompt"],
            parameters=LLMParameters(**json.loads(row["parameters"])),
            response=row["response"],
            rating=row["rating"],
            feedback=row["feedback"],
            experiment_type=row["experiment_type"],
            name=row["name"]
        )
        experiments.append(experiment)

    conn.close()
    return experiments


def get_experiment_by_id(experiment_id: int) -> Experiment | None:
    """Get a single experiment by ID."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    experiment = Experiment(
        id=row["id"],
        created_at=row["created_at"],
        prompt=row["prompt"],
        parameters=LLMParameters(**json.loads(row["parameters"])),
        response=row["response"],
        rating=row["rating"],
        feedback=row["feedback"],
        experiment_type=row["experiment_type"],
        name=row["name"]
    )

    conn.close()
    return experiment


def get_comparison_results(experiment_id: int) -> list[ComparisonResult]:
    """Get all comparison results for an experiment."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM comparison_results
        WHERE experiment_id = ?
        ORDER BY prompt_index
    """, (experiment_id,))

    rows = cursor.fetchall()
    results = []

    for row in rows:
        result = ComparisonResult(
            id=row["id"],
            experiment_id=row["experiment_id"],
            prompt_index=row["prompt_index"],
            response=row["response"],
            rating=row["rating"]
        )
        results.append(result)

    conn.close()
    return results


def update_experiment_rating(experiment_id: int, rating: int, feedback: str = None):
    """Update the rating and feedback for an experiment."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE experiments
        SET rating = ?, feedback = ?
        WHERE id = ?
    """, (rating, feedback, experiment_id))

    conn.commit()
    conn.close()


def update_comparison_result_rating(result_id: int, rating: int):
    """Update the rating for a comparison result."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE comparison_results
        SET rating = ?
        WHERE id = ?
    """, (rating, result_id))

    conn.commit()
    conn.close()


def delete_experiment(experiment_id: int):
    """Delete an experiment and its comparison results."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete comparison results first (foreign key)
    cursor.execute("DELETE FROM comparison_results WHERE experiment_id = ?", (experiment_id,))
    cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

    conn.commit()
    conn.close()


def get_experiment_count() -> int:
    """Get total number of experiments."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM experiments")
    count = cursor.fetchone()[0]

    conn.close()
    return count
