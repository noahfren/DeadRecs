# Roadmap

Future improvements planned for DeadRecs.

## Natural Language Queries

Allow users to describe what they're looking for in plain English instead of requiring exact show dates or song titles.

For example:
- "Find me a high-energy 1977 show with a long Scarlet/Fire"
- "Something like Cornell '77 but from the 80s"
- "Shows where drums/space is especially good"

This would involve interpreting free-text input, mapping it to the existing graph structure, and returning relevant recommendations.

## Web Application

Host DeadRecs as a server with a web frontend so users can get recommendations from a browser instead of the CLI. This would include:

- A backend API (e.g., FastAPI) exposing the recommendation engine over HTTP
- A web UI for browsing recommendations, exploring the graph, and discovering shows
- The ability to share recommended shows or setlists via links

## Evaluation Benchmark

Create a benchmark suite to evaluate the recommendation model's performance across updates. This would include:

- A curated set of test queries with known-good recommendations (ground truth)
- Quantitative metrics (e.g., precision@k, recall@k, NDCG) to measure recommendation quality
- Automated comparison of results across model versions to detect regressions or improvements

## Additional Data Sources

Research and integrate additional data sources beyond Headyversion.com to improve model quality. Potential sources include:

- Setlist.fm or other setlist databases for more complete show/song metadata
- Archive.org for audience recordings, show metadata, and listener engagement signals
- Deadhead community forums or Reddit for qualitative ratings and discussion sentiment
- Jerry Garcia Band, Phil & Friends, and other related project data to expand the graph
