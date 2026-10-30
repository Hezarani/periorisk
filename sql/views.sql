-- Summary views on top of the tables written by the pipeline.
-- DROP + CREATE works on both SQLite and PostgreSQL.
DROP VIEW IF EXISTS v_prevalence_by_cycle;
CREATE VIEW v_prevalence_by_cycle AS
SELECT cycle,
       COUNT(*) AS participants,
       SUM(perio_mod_severe) AS cases,
       ROUND(AVG(perio_mod_severe) * 100, 1) AS crude_prevalence_pct
FROM features
GROUP BY cycle;
DROP VIEW IF EXISTS v_case_definition_counts;
CREATE VIEW v_case_definition_counts AS
SELECT perio_category, COUNT(*) AS participants
FROM clean_cohort
GROUP BY perio_category
