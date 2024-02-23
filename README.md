# ffwrapped
This is meant to be a large multi-faceted project that will seek to provide both:
- Fantasy football related analysis and research
- Tools and utilities for analysis

## 1. ffsimquery
Our very first service is live now! Check it out!
- [Repo](src/modules/simQuery)
- [README](src/modules/simQuery/ffsimquery.md)

### A. Current architecture
1. Backend
   - Application
     - Backend Flask app with two POST endpoints
     - Deployed via AWS App Runner
   - Data source
     - ~500 MB data in *.csv files
     - Saved directly in container with source code
2. CLI Front-end
   - Simple, CLI-like tool
     - Implemented as Python module
     - Calls backend at AWS-assigned default domain for App Runner-deployed service

### B. Next steps
1. Get more data
   - Decide: where stored, how queried?
     - RDB
     - S3
     - S3 + ECS ephemeral storage
   - Set up resources
   - Set up remote job to populate data source
   - Refactor backend-application code accordingly  
2. Expand features of tool
   - Allow selection at any point in draft, not just sequential
   - Allow other teams to "draft", rule out drafted players from table
3. More accessible UI
   - Package/containerize UI as well, somehow
   - Full-blown frontend would be ideal, but don't know anything about FE
4. Move back-end app from App Runner to ECS

## 2. Future feature ideas
**Attribution Analysis:** How much of team performance is draft vs starter selection?

**Research Topic:** Do teams outperform each other on draft or starter selection?

**Actual Draft Tool:** Train an ML-based draft tool using RL? Make live draft recommendations, like more robust simquery

