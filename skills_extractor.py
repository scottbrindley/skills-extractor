import requests
import re
import spacy
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from huggingface_hub import snapshot_download, login
from collections import Counter


def fetch_linkedin_job_listings(start=0):
    """
    Fetches batches of job listings from LinledIn, starting from a given API index (start)

    Parameters
    ----------
    start : int
        Specifies starting index for fetching from API

    Returns
    -------
    dict
        A dictionary of job listings containing:
           - title: str
           - url: str
           - location: str

    Notes
    -----
    - Any job listing without one of the following terms in the title is discarded: "Data", "BI", "Modeler"
    """

    url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    payload = {
        "keywords": "data engineer bi",
        "location": "Australia",
        "start": str(start),
    }

    response = requests.get(url, params=payload)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    # Loop through each <li>
    for li in soup.find_all("li"):

        # Extract job title
        title_tag = li.find("h3", class_="base-search-card__title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Extract job URL
        link_tag = li.find("a", class_="base-card__full-link")
        url = link_tag["href"] if link_tag and link_tag.has_attr("href") else None

        # Extract location
        loc_tag = li.find("span", class_="job-search-card__location")
        location = loc_tag.get_text(strip=True) if loc_tag else None

        # Filter results to only include jobs with "Data", "BI", or "Modeler" in the title (case insensitive)
        if title and any(keyword in title.lower() for keyword in ("data", "bi", "modeler")):
            results.append({"title": title, "url": url, "location": location})

    return results


def extract_job_id_from_linkedin_url(url):
    """
    Extracts job ID from LinkedIn job page URL

    Parameters
    ----------
    url : str
        The LinkedIn job page URL

    Returns
    -------
    str or None
        The extracted job ID or None if extraction fails
    """

    try:

        parsed = urlparse(url)
        # Extract the last part of the path
        last_segment = parsed.path.rsplit("/", 1)[-1]
        # Remove anything after '?'
        job_id_str = last_segment.split("?", 1)[0]
        # Extract the number at the end
        match = re.search(r"(\d+)$", job_id_str)
        job_id = match.group(1) if match else None

        return job_id

    except TypeError:
        print(f"Invalid URL: {url}")
        
        return None


def fetch_linkedin_job_descriptions(jobs):
    """
    Fetch job descriptions for each job listing using LinkedIn HTML Response

    Parameters
    ----------
    jobs : list
        A list of job listings, where each listing is a dictionary containing:
           - title: str
           - url: str
           - location: str

    Returns
    -------
    list
        A list of job listings, where each listing is a dictionary containing:
           - title: str
           - url: str
           - location: str
           - description: str
    """

    for job in jobs:

        job_id = extract_job_id_from_linkedin_url(job["url"])

        if job_id is not None:

            url = f" https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"

            response = requests.get(url)
            print(f"Fetched job description for job ID {job_id} with status code {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the div containing the job description
            desc_div = soup.find(
                "div", class_="description__text description__text--rich"
            )

            # Extract text (preserves line breaks)
            job_description = (
                desc_div.get_text(separator="\n", strip=True) if desc_div else None
            )

        else:
            print(f"Could not extract job ID from URL: {job['url']}")
            job_description = ""

        # Add the description to the job dictionary
        job["description"] = job_description

    return jobs


def extract_skills_from_description(jobs, nlp):
    """
    Extract skills (named entities) from job description using a Hugging Face
    NLP NER model fine-tuned for skill extraction

    Parameters
    ----------
    jobs : list
        A list of job listings, where each listing is a dictionary containing:
           - title: str
           - url: str
           - location: str
           - description: str
    Returns
    -------
    list
        A list of job listings, where each listing is a dictionary containing:
           - title: str
           - url: str
           - location: str
           - description: str
           - skills: dict[str, list[str]]
                A dictionary of extracted skills, where the key is the entity label (e.g. "SKILLS") 
                and the value is a list of unique skill names extracted from the description
    """

    for job in jobs:

        text = job["description"]
        if text is not None:
            doc = nlp(text)

        skills = {}
        for ent in doc.ents:
            if ent.text not in skills.get(ent.label_, []):
                skills.setdefault(ent.label_, []).append(ent.text)

        job["skills"] = skills

    return jobs


def calculate_skill_frequencies(jobs_dict):
    """
    Derive the top N most in-demand skills

    Parameters
    ----------
    jobs : list
        A list of job listings, where each listing is a dictionary containing:
           - title: str
           - url: str
           - location: str
           - description: str
           - skills: dict[str, list[str]]

    Returns
    -------
    dict
        - skill_counts: dict[str, int]
    """
    
    all_skills = []

    for job in jobs_dict:
        skills = job["skills"].get("SKILLS", [])
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    top_n = 20
    top_skills = dict(skill_counts.most_common(top_n))

    return {"skill_counts": top_skills}


def extract_skills():

    # Connect to Hugging Face Hub and download the NLP model
    login(token=os.getenv("HF_TOKEN"))

    # Download NLP model from the Hub
    print(f"Downloading NLP model...")
    model_path = snapshot_download("amjad-awad/skill-extractor", repo_type="model")

    # Load the model with spaCy
    print(f"Loading NLP model...")
    nlp = spacy.load(model_path)

    jobs_dict = []
    # Assume there are no more than 500 job listings
    # We want to fetch in batches of 10, as that's what the LinkedIn API seems to return
    limit = 10
    for page in range(0, limit, 10):

        # Fetch job listings and the job description for each listing
        print(f"Fetching jobs starting from index {page}...")
        results = fetch_linkedin_job_listings(page)
        results = fetch_linkedin_job_descriptions(results)
        results = extract_skills_from_description(results, nlp)

        # Add the results to the main jobs dictionary
        jobs_dict.extend(results)

    skill_counts = calculate_skill_frequencies(jobs_dict)
    print(skill_counts)
    return skill_counts


if __name__ == "__main__":
    extract_skills()
