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
