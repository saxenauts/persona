# Changelog

## [0.2.0] - 2025-06-19

### Major Features & Improvements
- Comprehensive logging and error handling:
  - Centralized logging configuration with structured log format
  - All print statements replaced with proper logging (DEBUG, INFO, WARNING, ERROR)
  - FastAPI error handling: meaningful HTTP status codes, input validation, user existence checks, sanitized error messages
  - Database connection (503) and external service (502) error handling
- API and service refactors:
  - Standardized on `UnstructuredData` model for all ingestion and API flows
  - RESTful API conventions: user_id as path parameter, no user_id in request bodies
  - Dependency injection for `GraphOps` (global app state, FastAPI DI)
  - Idempotent user creation, improved user deletion, and data integrity
  - Updated all tests and documentation for new API patterns
- Documentation:
  - Expanded developer and onboarding docs
  - API usage, troubleshooting, and architecture guides
  - Updated README and codebase overview

### Bug Fixes
- Fixed double node creation in GraphOps
- Fixed property/perspective population in NodeModel
- Fixed user deletion in Neo4j
- Fixed example script for local and Docker environments

### Quality Assurance
- 100% test coverage, all tests passing
- Zero breaking changes, full backward compatibility

## [0.1.1] - 2024-08-30
- Added custom instructions to the app
- Added example.ipynb to the repo

## [0.1.0] - 2024-08-09
- Initial release
- Basic user management
- Unstructured data ingestion
- Graph generation and graph context retrieval
- RAG query