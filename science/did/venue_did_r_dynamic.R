#!/usr/bin/env Rscript
# ==============================================================================
# run_did.R — Callaway & Sant'Anna DiD via R's `did` package
#
# Usage from command line:
#   Rscript run_did.R \
#     --input data.csv \
#     --outcome cum_citations_na \
#     --output_dir results/ \
#     --n_boot 99 \
#     --min_venue_year 1930 \
#     --min_cohort_size 10
#
# Output: CSV files for dynamic, cohort, and overall ATT estimates
# ==============================================================================

suppressPackageStartupMessages({
  library(did)
  library(optparse)
  library(data.table)
})

# --- CLI arguments ---
option_list <- list(
  make_option("--input", type = "character", help = "Path to semicolon-delimited CSV"),
  make_option("--outcome", type = "character", default = "cum_citations_na",
              help = "Outcome variable name"),
  make_option("--output_dir", type = "character", default = "did_results",
              help = "Output directory"),
  make_option("--label", type = "character", default = "",
              help = "Label for output files (e.g., 'Nature_citations')"),
  make_option("--n_boot", type = "integer", default = 99,
              help = "Number of bootstrap iterations"),
  make_option("--est_method", type = "character", default = "reg",
              help = "Estimation method: reg, ipw, or dr"),
  make_option("--min_venue_year", type = "integer", default = 1930,
              help = "Drop venue cohorts before this year"),
  make_option("--min_cohort_size", type = "integer", default = 10,
              help = "Minimum treated authors per venue year cohort"),
  make_option("--covariates", type = "character", default = "",
              help = "Comma-separated covariate names, or 'none' for unconditional"),
  make_option("--subgroup_col", type = "character", default = "",
              help = "Column to split subgroups on (e.g., 'gender_label')"),
  make_option("--subgroup_val", type = "character", default = "",
              help = "Value to filter subgroup_col on (e.g., 'female')")
)

opt <- parse_args(OptionParser(option_list = option_list))

cat("\n==================================================\n")
cat("  R did package — Callaway & Sant'Anna estimation\n")
cat("==================================================\n")

# --- Load data ---
cat("\nLoading:", opt$input, "\n")
df <- fread(opt$input, sep = ";")
cat("  Rows:", nrow(df), " Authors:", uniqueN(df$author_id), "\n")

# --- Create numeric author ID ---
df[, author_int_id := as.integer(factor(author_id))]

# --- Treated indicator ---
treated_ids <- unique(df[is_venue == 1, author_id])
df[, is_treated := as.integer(author_id %in% treated_ids)]

# --- panel_time = calendar year ---
df[, panel_time := as.integer(year)]

# --- first_treat = venue_year for treated, 0 for controls ---
df[, first_treat := 0L]
df[is_treated == 1, first_treat := as.integer(venue_year)]

# Fix bad treated
df[is_treated == 1 & (is.na(first_treat) | first_treat <= 0),
   `:=`(is_treated = 0L, first_treat = 0L)]

# --- Filter 1: venue year cutoff ---
old_ids <- unique(df[first_treat > 0 & first_treat < opt$min_venue_year, author_id])
if (length(old_ids) > 0) {
  df <- df[!(author_id %in% old_ids)]
  cat("  Dropped", length(old_ids), "authors with venue_year <", opt$min_venue_year, "\n")
}

# --- Filter 2: minimum cohort size ---
cohort_sizes <- df[first_treat > 0, .(n = uniqueN(author_id)), by = first_treat]
small_cohorts <- cohort_sizes[n < opt$min_cohort_size, first_treat]
if (length(small_cohorts) > 0) {
  small_ids <- unique(df[first_treat %in% small_cohorts, author_id])
  df <- df[!(author_id %in% small_ids)]
  cat("  Dropped", length(small_cohorts), "cohorts with <", opt$min_cohort_size,
      "treated (", length(small_ids), "authors)\n")
}

# --- Filter 3: trim calendar years ---
valid_cohorts <- unique(df[first_treat > 0, first_treat])
if (length(valid_cohorts) > 0) {
  trim_min <- min(valid_cohorts) - 6
  trim_max <- max(valid_cohorts) + 11
  df <- df[panel_time >= trim_min & panel_time <= trim_max]
  cat("  Trimmed calendar years to [", trim_min, ",", trim_max, "]\n")
}

# --- Subgroup filter (if requested) ---
if (opt$subgroup_col != "" && opt$subgroup_val != "") {
  cat("  Filtering subgroup:", opt$subgroup_col, "=", opt$subgroup_val, "\n")
  # Keep controls + treated authors matching the subgroup
  # For controls, we keep all (they might not have the subgroup attribute)
  keep_ids <- unique(df[
    (is_treated == 0) |
    (is_treated == 1 & get(opt$subgroup_col) == opt$subgroup_val),
    author_id
  ])
  df <- df[author_id %in% keep_ids]
}

# --- Fill NAs in outcome ---
if (!(opt$outcome %in% names(df))) {
  stop(paste("Outcome", opt$outcome, "not found in data"))
}
df[is.na(get(opt$outcome)), (opt$outcome) := 0]

# --- Remove rows with NA panel_time ---
df <- df[!is.na(panel_time)]

# --- Summary ---
n_treated <- uniqueN(df[is_treated == 1, author_id])
n_control <- uniqueN(df[is_treated == 0, author_id])
cat("\n  Treated:", n_treated, " Control:", n_control, "\n")
cat("  Calendar year range:", min(df$panel_time), "-", max(df$panel_time), "\n")
cat("  Cohorts:", length(unique(df[first_treat > 0, first_treat])), "\n")
cat("  Outcome:", opt$outcome, "\n")

# --- Build covariate formula ---
if (opt$covariates == "" || opt$covariates == "none") {
  xformla <- as.formula(paste0("~1"))
  cat("  Covariates: unconditional (~1)\n")
} else {
  cov_list <- trimws(unlist(strsplit(opt$covariates, ",")))
  cov_list <- cov_list[cov_list %in% names(df)]
  if (length(cov_list) == 0) {
    xformla <- as.formula("~1")
    cat("  Covariates: none found, using unconditional (~1)\n")
  } else {
    xformla <- as.formula(paste0("~", paste(cov_list, collapse = "+")))
    cat("  Covariates:", deparse(xformla), "\n")
  }
}

# --- Run att_gt ---
cat("\n  Running att_gt()...\n")
t0 <- proc.time()

out <- tryCatch({
  att_gt(
    yname = opt$outcome,
    gname = "first_treat",
    idname = "author_int_id",
    tname = "panel_time",
    xformla = xformla,
    data = as.data.frame(df),
    control_group = "nevertreated",
    panel = FALSE,
    est_method = opt$est_method,
    bstrap = TRUE,
    biters = opt$n_boot,
    print_details = FALSE
  )
}, error = function(e) {
  cat("  ERROR in att_gt():", conditionMessage(e), "\n")
  cat("  Trying unconditional (~1)...\n")
  att_gt(
    yname = opt$outcome,
    gname = "first_treat",
    idname = "author_int_id",
    tname = "panel_time",
    xformla = ~1,
    data = as.data.frame(df),
    control_group = "nevertreated",
    panel = FALSE,
    est_method = opt$est_method,
    bstrap = TRUE,
    biters = opt$n_boot,
    print_details = FALSE
  )
})

elapsed <- (proc.time() - t0)[3]
cat("  att_gt() completed in", round(elapsed, 1), "seconds\n")

# --- Create output directory ---
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

# --- Helper: build label for filenames ---
file_prefix <- if (opt$label != "") opt$label else gsub("[^a-zA-Z0-9_]", "_", opt$outcome)
if (opt$subgroup_col != "" && opt$subgroup_val != "") {
  file_prefix <- paste0(file_prefix, "_", opt$subgroup_col, "_", opt$subgroup_val)
}

# --- Extract ATT(g,t) ---
attgt_df <- data.frame(
  group = out$group,
  time = out$t,
  att = out$att,
  se = out$se,
  ci_lower = out$att - 1.96 * out$se,
  ci_upper = out$att + 1.96 * out$se,
  pvalue = 2 * pnorm(-abs(out$att / out$se))
)
attgt_path <- file.path(opt$output_dir, paste0("attgt_", file_prefix, ".csv"))
write.csv(attgt_df, attgt_path, row.names = FALSE)
cat("  Saved ATT(g,t):", attgt_path, "(", nrow(attgt_df), "rows)\n")

# --- Dynamic aggregation (event-study) ---
tryCatch({
  agg_dyn <- aggte(out, type = "dynamic")
  dyn_df <- data.frame(
    event_time = agg_dyn$egt,
    att = agg_dyn$att.egt,
    se = agg_dyn$se.egt,
    ci_lower = agg_dyn$att.egt - 1.96 * agg_dyn$se.egt,
    ci_upper = agg_dyn$att.egt + 1.96 * agg_dyn$se.egt,
    pvalue = 2 * pnorm(-abs(agg_dyn$att.egt / agg_dyn$se.egt))
  )
  dyn_path <- file.path(opt$output_dir, paste0("dynamic_", file_prefix, ".csv"))
  write.csv(dyn_df, dyn_path, row.names = FALSE)
  cat("  Saved dynamic:", dyn_path, "(", nrow(dyn_df), "rows)\n")

  # Print summary
  cat("\n  Dynamic ATT (event-study):\n")
  for (i in seq_len(nrow(dyn_df))) {
    sig <- if (!is.na(dyn_df$pvalue[i]) && dyn_df$pvalue[i] < 0.05) "*" else ""
    cat(sprintf("    e=%3d: ATT=%10.2f (SE=%.2f) p=%.4f%s\n",
                dyn_df$event_time[i], dyn_df$att[i], dyn_df$se[i],
                dyn_df$pvalue[i], sig))
  }
}, error = function(e) {
  cat("  WARNING: aggte('dynamic') failed:", conditionMessage(e), "\n")
})

# --- Group/cohort aggregation ---
tryCatch({
  agg_grp <- aggte(out, type = "group")
  grp_df <- data.frame(
    group = agg_grp$egt,
    att = agg_grp$att.egt,
    se = agg_grp$se.egt,
    ci_lower = agg_grp$att.egt - 1.96 * agg_grp$se.egt,
    ci_upper = agg_grp$att.egt + 1.96 * agg_grp$se.egt,
    pvalue = 2 * pnorm(-abs(agg_grp$att.egt / agg_grp$se.egt))
  )
  grp_path <- file.path(opt$output_dir, paste0("cohort_", file_prefix, ".csv"))
  write.csv(grp_df, grp_path, row.names = FALSE)
  cat("  Saved cohort:", grp_path, "(", nrow(grp_df), "rows)\n")
}, error = function(e) {
  cat("  WARNING: aggte('group') failed:", conditionMessage(e), "\n")
})

# --- Simple/overall aggregation ---
tryCatch({
  agg_simple <- aggte(out, type = "simple")
  overall_df <- data.frame(
    att = agg_simple$overall.att,
    se = agg_simple$overall.se,
    ci_lower = agg_simple$overall.att - 1.96 * agg_simple$overall.se,
    ci_upper = agg_simple$overall.att + 1.96 * agg_simple$overall.se,
    pvalue = 2 * pnorm(-abs(agg_simple$overall.att / agg_simple$overall.se))
  )
  ovr_path <- file.path(opt$output_dir, paste0("overall_", file_prefix, ".csv"))
  write.csv(overall_df, ovr_path, row.names = FALSE)
  cat("  Saved overall:", ovr_path, "\n")
  cat(sprintf("\n  Overall ATT: %.2f (SE=%.2f, p=%.4f)\n",
              overall_df$att, overall_df$se, overall_df$pvalue))
}, error = function(e) {
  cat("  WARNING: aggte('simple') failed:", conditionMessage(e), "\n")
})

cat("\n  Done!\n\n")
