"""
Google Play Store Analytics – SQL Script Executor & Validator
=============================================================
Runs validate_etl.sql and analytics_queries.sql against the Google_Play
database, reports pass/fail per query, and summarises data-quality findings.

Usage:
    python run_sql_tests.py --password YOUR_MYSQL_PASSWORD
"""

import os
import argparse
import re
import sys
import pymysql
from pymysql.cursors import DictCursor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run SQL validation & analytics scripts.")
    parser.add_argument("--host", default=os.getenv("DB_HOST", "localhost"), help="MySQL database host")
    parser.add_argument("--port", type=int, default=int(os.getenv("DB_PORT", "3306")), help="MySQL database port")
    parser.add_argument("--user", default=os.getenv("DB_USER", "root"), help="MySQL database username")
    parser.add_argument("--password", default=os.getenv("DB_PASSWORD", ""), help="MySQL database password")
    parser.add_argument("--database", default=os.getenv("DB_NAME", "Google_Play"), help="MySQL database name")
    return parser.parse_args()


def extract_queries(filepath):
    """
    Split a SQL file into individual queries, preserving the comment
    immediately before each query as its label.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the USE statement (handled at connection level)
    content = re.sub(r'(?mi)^\s*USE\s+\w+\s*;\s*$', '', content)

    # Split on semicolons (but not inside strings – good-enough heuristic)
    raw_blocks = content.split(";")

    queries = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        # Skip pure-comment blocks with no SQL
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        sql_lines = [l for l in lines if not l.startswith("--")]
        if not sql_lines:
            continue

        # Extract the last comment block before the SQL as the label
        comment_lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                comment_lines.append(stripped.lstrip("- ").strip())
            elif stripped:
                break

        label = " | ".join(comment_lines[-2:]) if comment_lines else "Unlabelled query"
        queries.append({"label": label, "sql": block + ";"})

    return queries


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Connect ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("Google Play Store Analytics – SQL Test Runner")
    print("=" * 72)

    try:
        conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
            cursorclass=DictCursor,
            connect_timeout=10,
        )
        print(f"\n✅  Connected to {args.database} on {args.host}:{args.port}\n")
    except Exception as e:
        print(f"\n❌  Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Verify schema ────────────────────────────────────────────────────
    print("-" * 72)
    print("SCHEMA VERIFICATION")
    print("-" * 72)
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = [list(r.values())[0] for r in cur.fetchall()]
        print(f"Tables found: {tables}\n")

        for tbl in tables:
            cur.execute(f"DESCRIBE `{tbl}`")
            cols = [r["Field"] for r in cur.fetchall()]
            cur.execute(f"SELECT COUNT(*) AS cnt FROM `{tbl}`")
            cnt = cur.fetchone()["cnt"]
            print(f"  {tbl:30s}  cols={len(cols):>2d}  rows={cnt:>8,d}")
            print(f"    └─ {', '.join(cols)}")
        print()

    # ── Execute scripts ──────────────────────────────────────────────────
    scripts = [
        ("sql/validate_etl.sql", "VALIDATION"),
        ("sql/analytics_queries.sql", "ANALYTICS"),
    ]

    total_pass = 0
    total_fail = 0
    total_queries = 0
    fixes_applied = []
    data_quality_issues = []

    for filepath, section_name in scripts:
        print("=" * 72)
        print(f"EXECUTING: {filepath}  ({section_name})")
        print("=" * 72)

        queries = extract_queries(filepath)
        section_pass = 0
        section_fail = 0

        for idx, q in enumerate(queries, 1):
            total_queries += 1
            label = q["label"][:90]
            sql = q["sql"]

            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    row_count = len(rows)

                status = "PASS"
                section_pass += 1
                total_pass += 1

                # ── Data quality inspection (validation script only) ─────
                if section_name == "VALIDATION" and rows:
                    for row in rows:
                        for key, val in row.items():
                            if "violation" in key.lower() or "violation" in str(row.get("validation", "")):
                                if val is not None and isinstance(val, (int, float)) and val > 0:
                                    issue = f"{row.get('validation', label)}: {val}"
                                    data_quality_issues.append(issue)

                print(f"  [{status}] Q{total_queries:>02d}  ({row_count:>5d} rows)  {label}")

            except Exception as e:
                status = "FAIL"
                section_fail += 1
                total_fail += 1
                err_msg = str(e).split("\n")[0]
                print(f"  [{status}] Q{total_queries:>02d}  ✗ ERROR     {label}")
                print(f"           └─ {err_msg}")

        print(f"\n  Section result: {section_pass} passed, {section_fail} failed "
              f"out of {section_pass + section_fail}\n")

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("EXECUTION SUMMARY")
    print("=" * 72)
    print(f"  Total queries executed : {total_queries}")
    print(f"  Passed                 : {total_pass}")
    print(f"  Failed                 : {total_fail}")
    print(f"  Fixes applied          : {len(fixes_applied)}")
    if fixes_applied:
        for fix in fixes_applied:
            print(f"    • {fix}")

    print(f"\n  Data quality issues found : {len(data_quality_issues)}")
    if data_quality_issues:
        for issue in data_quality_issues:
            print(f"    ⚠  {issue}")

    print()
    if total_fail == 0 and not fixes_applied:
        print("✅  SUCCESS – All queries executed successfully.")
    elif total_fail == 0 and fixes_applied:
        print("⚠️  SUCCESS WITH FIXES – Executed after corrections (see above).")
    else:
        print("❌  FAILED – See errors above.")
    print("=" * 72)

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
