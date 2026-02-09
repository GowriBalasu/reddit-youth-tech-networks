Tech Talks: Exploring Hidden User Communities in Teen, Parenting, and Technology-Oriented Subreddits

**Network analysis of cross-community user behavior across youth, parenting, and technology subreddits on Reddit.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project uses network analysis and community detection to explore how users in teen-focused, parenting-focused, and technology-focused subreddit communities interact, overlap, and discuss technology. The analysis scraped ~400K interactions from 40+ subreddits, constructed co-comment interaction networks, and applied the Louvain community detection algorithm to identify hidden cross-community structures.

**Key Findings:**
- Analyzed **179,886 unique users** and **393,410 co-comment interactions** across 3 subreddit categories
- Identified **893 distinct communities** (modularity = 0.8) in the full network, indicating highly segregated interaction patterns
- Cross-posting subgraph of **2,034 overlapping users** revealed **65 communities** with more meaningful mixing across categories
- Thematic analysis of cross-community content surfaced shared concerns around social connection, mediated through technology

## Research Questions

1. How do users in teen-focused and adult-focused subreddit categories interact with users in technology-focused categories?
2. How and why do interactions and community topics on technology differ or converge across youth and parent populations?

## Motivation

Understanding how different age groups engage with technology discourse online has direct implications for:
- **AI safety & age-appropriate design**: Informing how AI systems should adapt behavior for different user populations
- **Platform governance**: Understanding cross-demographic information flow on social platforms
- **Digital parenting research**: Identifying gaps between parent and teen perspectives on technology

## Repository Structure

```
├── scripts/
│   ├── 01_data_collection.py      # Reddit data scraping via Pushshift API
│   ├── 02_network_construction.py # Building co-comment interaction networks
│   ├── 03_community_detection.py  # Louvain algorithm & cross-posting subgraph
│   ├── 04_thematic_analysis.py    # Content analysis of community text data
│   └── 05_visualization.py        # Network visualization & statistics
├── data/
│   └── subreddit_config.json      # Subreddit categories and metadata
├── visualizations/                # Network graphs and figures
├── docs/
│   └── project_report.pdf         # Full academic report
├── requirements.txt
└── README.md
```

## Methods

### Data Collection
- **Source**: Reddit via Pushshift API
- **Time period**: April 2022 – April 2023
- **Scope**: 40+ subreddits across 3 categories (youth-focused, parenting-focused, technology-focused)
- **Result**: 22,174 submissions and 379,423 comments

### Network Construction
- **Model**: Co-comment interaction network (adapted from Hamilton et al., 2017 / Stanford SNAP)
- **Nodes**: Unique Reddit users (179,886)
- **Edges**: Co-comment thread interactions (393,410)

### Analysis
- **Community Detection**: Louvain algorithm (via NetworkX / Gephi)
- **Subgraph Analysis**: Filtered to users active in 2+ subreddit categories (2,034 users)
- **Content Analysis**: Open coding and thematic analysis of top community posts

## Technical Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Data collection, processing, analysis |
| Pushshift API | Historical Reddit data retrieval |
| NetworkX | Graph construction and analysis |
| python-louvain | Community detection |
| Gephi | Large-scale network visualization |
| pandas | Data cleaning and manipulation |
| matplotlib / seaborn | Statistical visualization |

## Key Results

| Metric | Full Network | Cross-Posting Subgraph |
|--------|-------------|----------------------|
| Nodes | 179,886 | 2,034 |
| Edges | 393,410 | — |
| Communities | 893 | 65 |
| Modularity | 0.80 | 0.70 |
| Users in 2 categories | 2,014 | — |
| Users in all 3 categories | 20 | — |

## Connection to Broader Research

This project is part of a broader research agenda on **youth-centered AI design** and **age-appropriate technology governance**. The network analysis methodology serves as a computational complement to qualitative interview and focus group research with young people (ages 11–17) on their perceptions of AI systems.

Related work:
- Balasubramaniam, G. et al. (2024). *[CHI 2024 publication on surveillance technology]*
- Dissertation: *"Seen, Not Surveilled: Youth-Centered Perspectives on Age-Appropriate Artificial Intelligence"* (in progress)

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/[your-username]/reddit-youth-tech-networks.git
cd reddit-youth-tech-networks

# Install dependencies
pip install -r requirements.txt

# Run the analysis pipeline
python scripts/01_data_collection.py
python scripts/02_network_construction.py
python scripts/03_community_detection.py
python scripts/04_thematic_analysis.py
python scripts/05_visualization.py
```

> **Note**: The Pushshift API has been deprecated as of 2023. The data collection script is preserved for methodological documentation. Contact the author for access to the processed dataset.

## References

- Borgatti, S. P., Everett, M. G., & Johnson, J. C. (2013). *Analyzing Social Networks*. SAGE.
- Hamilton, W. L., Zhang, J., Danescu-Niculescu-Mizil, C., Jurafsky, D., & Leskovec, J. (2017). Loyalty in Online Communities. *WWW 2017*.
- Lupton, D., & Williamson, B. (2017). The datafied child. *New Media & Society*, 19(5), 780–794.
- Stoilova, M., Nandagiri, R., & Livingstone, S. (2021). Children's understanding of personal data and privacy online. *Information, Communication & Society*, 24(4), 557–575.

## Author

**Gowri Balasubramaniam**  
PhD Candidate, School of Information Sciences, University of Illinois Urbana-Champaign  
Research: Youth-centered AI design, privacy governance, age-appropriate technology  

## License

CC BY-NC 4.0
