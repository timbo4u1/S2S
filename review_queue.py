"""
S2S Review Queue — BRONZE tier human review system
Auto-populated when adapters certify records as BRONZE (30-59).
Records stay in queue until Approve or Reject via review.html

Usage:
    from review_queue import ReviewQueue
    queue = ReviewQueue()
    queue.add(record, source_file="path/to/original.json")

CLI:
    python3 review_queue.py --stats
    python3 review_queue.py --flush-approved    # move approved → s2s_dataset/
    python3 review_queue.py --flush-rejected    # remove rejected from queue
"""

import json
import os
import time
import sys
import argparse
from pathlib import Path

QUEUE_FILE = "review_queue.json"


class ReviewQueue:
    def __init__(self, queue_file=QUEUE_FILE):
        self.queue_file = queue_file
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "1.0", "items": []}

    def _save(self):
        with open(self.queue_file, "w") as f:
            json.dump(self._data, f, indent=2)

    def add(self, record: dict, source_file: str = None):
        """Add a BRONZE record to the review queue."""
        tier = record.get("physics_tier", record.get("tier", ""))
        score = record.get("physics_score", record.get("physical_law_score", 0))

        item = {
            "id": f"bronze_{int(time.time() * 1000)}_{len(self._data['items'])}",
            "status": "pending",          # pending | approved | rejected
            "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reviewed_at": None,
            "reviewer_notes": "",
            "source_file": source_file,
            "record": record,
            # Denormalized for fast UI rendering
            "action": record.get("action", "unknown"),
            "domain": record.get("domain", "unknown"),
            "person_id": record.get("person_id", "unknown"),
            "dataset_source": record.get("dataset_source", "unknown"),
            "physics_tier": tier,
            "physics_score": score,
            "laws_passed": record.get("physics_laws_passed", record.get("laws_passed", [])),
            "laws_failed": record.get("physics_laws_failed", record.get("laws_failed", [])),
            "jerk_p95_ms3": record.get("jerk_p95_ms3", None),
            "imu_coupling_r": record.get("imu_coupling_r", None),
            "n_samples": record.get("n_samples", None),
            "duration_s": record.get("duration_s", None),
        }
        self._data["items"].append(item)
        self._save()
        return item["id"]

    def pending(self):
        return [i for i in self._data["items"] if i["status"] == "pending"]

    def approved(self):
        return [i for i in self._data["items"] if i["status"] == "approved"]

    def rejected(self):
        return [i for i in self._data["items"] if i["status"] == "rejected"]

    def review(self, item_id: str, decision: str, notes: str = ""):
        """Update a queue item's status. decision = 'approved' or 'rejected'."""
        assert decision in ("approved", "rejected")
        for item in self._data["items"]:
            if item["id"] == item_id:
                item["status"] = decision
                item["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                item["reviewer_notes"] = notes
                self._save()
                return True
        return False

    def flush_approved(self, dataset_dir="s2s_dataset"):
        """Move approved BRONZE records into the main dataset."""
        moved = 0
        for item in self.approved():
            record = item["record"]
            domain = record.get("domain", "UNKNOWN")
            action = record.get("action", "unknown")
            out_dir = Path(dataset_dir) / domain / action
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{action}_{item['person_id']}_bronze_approved_{item['id'][-8:]}.json"
            # Mark as human-reviewed
            record["human_reviewed"] = True
            record["reviewer_notes"] = item["reviewer_notes"]
            record["reviewed_at"] = item["reviewed_at"]
            with open(out_dir / fname, "w") as f:
                json.dump(record, f, separators=(",", ":"))
            moved += 1
        # Remove flushed items from queue
        self._data["items"] = [i for i in self._data["items"] if i["status"] != "approved"]
        self._save()
        return moved

    def flush_rejected(self):
        """Remove rejected records from queue permanently."""
        before = len(self._data["items"])
        self._data["items"] = [i for i in self._data["items"] if i["status"] != "rejected"]
        self._save()
        return before - len(self._data["items"])

    def stats(self):
        items = self._data["items"]
        pending = [i for i in items if i["status"] == "pending"]
        approved = [i for i in items if i["status"] == "approved"]
        rejected = [i for i in items if i["status"] == "rejected"]
        print(f"\nS2S Review Queue — {self.queue_file}")
        print(f"{'='*45}")
        print(f"  Total:    {len(items)}")
        print(f"  Pending:  {len(pending)}")
        print(f"  Approved: {len(approved)}")
        print(f"  Rejected: {len(rejected)}")
        if pending:
            print(f"\n  Pending by domain:")
            by_domain = {}
            for i in pending:
                by_domain[i["domain"]] = by_domain.get(i["domain"], 0) + 1
            for d, n in sorted(by_domain.items()):
                print(f"    {d:<20} {n}")
            print(f"\n  Score distribution:")
            scores = [i["physics_score"] for i in pending]
            buckets = {"30-39": 0, "40-49": 0, "50-59": 0}
            for s in scores:
                if s < 40: buckets["30-39"] += 1
                elif s < 50: buckets["40-49"] += 1
                else: buckets["50-59"] += 1
            for b, n in buckets.items():
                bar = "█" * n
                print(f"    {b}  {n:>4}  {bar}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="S2S Review Queue Manager")
    p.add_argument("--stats",          action="store_true", help="Show queue statistics")
    p.add_argument("--flush-approved", action="store_true", help="Move approved records to s2s_dataset/")
    p.add_argument("--flush-rejected", action="store_true", help="Remove rejected records from queue")
    p.add_argument("--dataset",        default="s2s_dataset", help="Dataset output dir")
    args = p.parse_args()

    q = ReviewQueue()

    if args.stats:
        q.stats()
    elif args.flush_approved:
        n = q.flush_approved(args.dataset)
        print(f"✓ Moved {n} approved BRONZE records to {args.dataset}/")
    elif args.flush_rejected:
        n = q.flush_rejected()
        print(f"✓ Removed {n} rejected records from queue")
    else:
        q.stats()
