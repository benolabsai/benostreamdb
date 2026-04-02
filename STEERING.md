# HyperStreamDB Steering Guidelines
*Based on the core tenets of The Pragmatic Programmer*

This file serves as the "steering" document for all development on HyperStreamDB. Every contribution, refactor, and release should align with these principles.

## 🛠️ The Craft of Programming
- **Care About Your Craft**: We are building a database. Reliability, performance, and correctness are not features; they are foundational.
- **Think! About Your Work**: Never program on autopilot. Every line of code, every SQL optimization, and every CI change should be done with intent.
- **Don't Live with Broken Windows**: If you see a compiler warning, a failing test, or a suboptimal query plan, fix it immediately. Technical debt is a debt with a high interest rate.

## 📐 Design & Orthogonality
- **DRY – Don't Repeat Yourself**: Avoid duplication in logic, data representation, and configuration. If the version is in the tag, don't hardcode it in three different files. 
- **ETC – Easy To Change**: Design components (like our SQL UDFs and storage layers) to be modular. A change in DataFusion versions should be a narrow edit, not a total rewrite.
- **Maintain Orthogonality**: Changes in the Python binding layer should not break the core Rust storage engine. Components should be independent and interchangeable.

## 🎯 Implementation Strategy
- **Use Tracer Bullets**: When implementing new features (like Windows build support), build a thin, end-to-end path first. Get the wheel building and uploading, then optimize the build speed.
- **Prototype to Learn**: Use benchmarks and tests to validate assumptions about GPU acceleration and indexing before committing to a final architecture.
- **Good Enough Software**: Don't let the pursuit of perfection prevent a stable release. Deliver reliable value (v0.1.2) and iterate.

## 🔍 Quality & Knowledge
- **Fix the Problem, Not Just the Symptom**: If a hybrid query fails, don't just hack the benchmark; understand why the parser failed and fix the underlying registration logic.
- **Invest in Your Knowledge Portfolio**: Keep the documentation (README, PROGRESS, BENCHMARK_FINDINGS) up to date. The codebase is only as good as the knowledge shared with it.
- **Critically Analyze Your Work**: Always question your own optimizations. Is the 10,000 rows/sec ingest rate truly efficient, or is there another "broken window" to fix?

## 🚀 Pragmatic Release Readiness
*Going forward, we'll automatically:*
- **Fix compiler warnings during implementation**: Resolve issues as they arise, never leave them as TODOs.
- **Add unit tests for new features**: Every new optimization or binding change requires a corresponding test.
- **Verify explain() and tests before tagging**: Run the full validation suite (including `benchmark_suite.py --quick`) before finalizing any `v0.X.Y` tag.
- **Properly attribute borrowed code**: Maintain licensing integrity by documenting borrowed patterns and optimizations.

---
*Stay Pragmatic. Stay Antigravity.*
