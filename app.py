from fastapi import FastAPI
from skills_extractor import extract_skills

app = FastAPI()


@app.get("/run")
def run():
    skill_counts = extract_skills()
    return {"status": "ok", "skill_counts": skill_counts}

@app.get("/test")
def run():
    return {"status": "ok", "message": "API called successfully"}
