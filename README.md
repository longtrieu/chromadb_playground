# IBM RAG and Agentic AI Professional Certificate - Course Materials

This GitHub repository contains the lab exercises, experiments, and assignments for the [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai) course.

### Setting up your development environment

Install chromadb and dependencies

```  bash
pip install chromadb==1.0.12
pip install sentence-transformers==4.1.0
```

### Run the search

``` bash
# Basic example
python3.11 similarity_search.py

# Advanced example
python3.11 similarity_employee_data.py

# Another advanced example
python3.11 books_advanced_search.py
```

### Key Observations for advanced example with Employee Data

#### Similarity Search Examples:

- Semantic Understanding: For the Python query, John Doe ranks highest (0.5156) because his skills explicitly include "Python, JavaScript, React, Node.js" - a perfect match for web development. Notice how the system found relevant candidates even though the query didn't exactly match the text.
- Cross-Department Results: The leadership search finds managers across different departments (Marketing, HR, Engineering), showing the system understands that "manager" roles exist in various contexts.
- Ranking Logic: Distance scores increase (0.5382 → 0.5467 → 0.5497) as semantic similarity decreases, demonstrating how Chroma DB quantifies relevance.
- Skills vs. Roles: Matthew Garcia appears in Python results despite his skills being "JavaScript, HTML/CSS" because his role as a "Software Engineer" is semantically related to development work.

#### Metadata Filtering Examples:

- Exact Filtering Power: The Engineering department filter returns exactly 8 out of 15 employees, showing that over half the workforce is in Engineering - a typical distribution for tech companies.
- Seniority Distribution: The 10+ years filter reveals 6 senior employees (40% of workforce), indicating a good mix of experience levels. Notice how roles reflect seniority: "Senior," "Lead," "Principal," and "Manager" titles.
- Role Hierarchy Visible: In the senior employees list, you can see the career progression: Software Engineer → Senior Software Engineer → Lead/Principal Engineer → Engineering Manager → Senior Architect.
- Geographic Concentration: Only 3 employees in California highlight the distributed nature of the workforce, with San Francisco having multiple employees as expected for a tech hub.
- Perfect Precision: Unlike similarity search, metadata filtering gives 100% precise results - every employee returned exactly matches the specified criteria.

#### Combined Search Results:

- Filtered Pool: From 15 total employees, only 4 meet both the semantic similarity AND the exact criteria (8+ years in major tech cities), demonstrating how filters narrow down results effectively.
- Semantic Ranking within Constraints: Michael Brown ranks highest (0.6726) because his Java/Spring Boot skills are semantically closest to "senior Python developer full-stack" among the filtered candidates.
- Interesting Discoveries: Chris Evans (Senior Architect) ranks second despite having "System design, distributed systems" skills rather than Python - showing the system understands that senior architects often have full-stack capabilities.
- Geographic + Experience Filtering: All results are from San Francisco, New York, or Seattle (major tech cities) and have 10+ years experience, proving the metadata filters worked perfectly.
- Real-World Relevance: This type of search mimics actual HR/recruiting scenarios where you need "senior developers with specific skills in key locations" - combining fuzzy skill matching with hard requirements.
- Distance Score Interpretation: Higher distances (0.8761 for Olivia) indicate she's still relevant to the query but with a looser semantic match, which is valuable for finding candidates with transferable skills.


#### Results:

- Distance scores represent semantic similarity, where lower values indicate higher similarity to the query.
- Metadata filtering acts as a hard constraint - only employees meeting these exact criteria are considered.
- Combined searches solve real business problems by balancing flexible skill matching with non-negotiable requirements.
- The system successfully demonstrates how vector databases can power sophisticated talent search platforms used by modern companies.
