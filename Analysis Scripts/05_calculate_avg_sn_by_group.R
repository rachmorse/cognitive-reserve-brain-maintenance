if (!require("dplyr")) {
  install.packages("dplyr")
  require("dplyr")
}
if (!require("tidyr")) {
  install.packages("tidyr")
  require("tidyr")
}

# Read in data and merge
betula_df <- read.csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Betula no outliers data wide.csv")
waha_df <- read.csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/WAHA no outliers data wide.csv")
cobra_df <- read.csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Cobra no outliers data wide.csv")
oslo_df <- read.csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Oslo no outliers data wide.csv")

# Start by making a merged df
merged_cohorts <- bind_rows(waha_df, cobra_df, betula_df, oslo_df)

# Filter to only include salience network and id 
sn_columns <- grep("^(AS)_.*\\.(AS)_.*\\.", colnames(merged_cohorts), value = TRUE)

filtered_data <- merged_cohorts %>% 
  select(id, study, sn_columns)

# Step 1: Reshape the data from wide to long, extracting ROI1, ROI2, and timepoint.
df_long <- filtered_data %>%
  pivot_longer(
    cols = matches("^AS_.*\\.AS_.*\\.(\\d+)$"),
    names_to = c("ROI1", "ROI2", "Timepoint"),
    names_pattern = "^AS_(.*?)\\.AS_(.*?)\\.(\\d+)$",
    values_to = "connectivity"
  ) %>%
  mutate(
    ROI1 = paste0("AS_", ROI1),
    ROI2 = paste0("AS_", ROI2)
  )

# Step 2: Convert ROI1 / ROI2 into a single ROI column, so each connectivity value is associated with both ROIs.
df_long_expanded <- df_long %>%
  pivot_longer(
    cols = c("ROI1", "ROI2"),
    names_to = "ROI_field",
    values_to = "ROI"
  )

# Step 3: Compute the average connectivity for each ROI at each timepoint per participant.
df_roi_time_participant <- df_long_expanded %>%
  group_by(id, study, ROI, Timepoint) %>%
  summarize(
    avg_connectivity_per_tp = mean(connectivity, na.rm = TRUE),
    .groups = "drop"
  )

# Step 4: Compute the final average across all timepoints, resulting in one value per ROI per participant.
df_roi_final_participant <- df_roi_time_participant %>%
  group_by(id, study, ROI) %>%
  summarize(
    ROI_av = mean(avg_connectivity_per_tp, na.rm = TRUE),
    .groups = "drop"
  )

sn_average_df <- df_roi_final_participant %>% 
  filter(!is.na(ROI_av))

# Pivot back to wide
df_wide <- sn_average_df %>%
  pivot_wider(
    id_cols    = c(id, study),
    names_from = ROI,
    values_from = ROI_av
  )

# Regress out the effect of study from each SN variable and create residual columns
sn_vars <- setdiff(names(df_wide), c("id", "study"))
for (var in sn_vars) {
  df_wide[[paste0(var, "_resid")]] <- NA
  non_na <- !is.na(df_wide[[var]]) & !is.na(df_wide$study)
  if (any(non_na)) {
    tmp_mod <- lm(df_wide[[var]] ~ as.factor(df_wide$study), data = df_wide, subset = non_na)
    df_wide[[paste0(var, "_resid")]][non_na] <- residuals(tmp_mod)
  }
}

# Keep only residual columns (and id)
resid_vars <- grep("_resid$", names(df_wide), value = TRUE)
df_wide <- df_wide %>%
  select(id, all_of(resid_vars))

# Scale the averages 
exclude_cols <- c("id")
cols_to_transform <- setdiff(names(df_wide), exclude_cols)
df_wide[cols_to_transform] <- scale(df_wide[cols_to_transform])

# Read in the CR data
clean_data <- read.csv("/tsd/p274/data/durable/projects/p027-cr_bm/Clean Data All Cohorts/Wide Format/Analysed data.csv")

# Create pathway variable 
clean_data <- clean_data %>%
  mutate(
    pathway = ifelse(dist_to_cr_line > dist_to_bm_line, "BM Pathway", "CR Pathway"),
  )

# Filter to only needed columns
cr_data <- clean_data %>%
  select(id, reverse_cr, pathway) 

# Merge
image_data <- merge(cr_data, df_wide, by = c("id"), all = TRUE) # This keeps all participants so need to drop NAs

# Filter for those on CR pathway
image_data <- image_data %>%
  filter(pathway == "CR Pathway") %>% 
  filter(!is.na(AS_L_ins_resid))

# Create CR groups
image_data$cr_group <- cut(image_data$reverse_cr,
                                breaks = quantile(image_data$reverse_cr, probs = seq(0, 1, (1 / 2)), na.rm = TRUE),
                                include.lowest = TRUE,
                                labels = c("Low CR", "High CR")
)

# Create a final df with one average value for each ROI divided by low and high CR group
df_summary <- image_data %>%
  group_by(cr_group) %>%
  summarize(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

write.csv(df_summary, file = "/tsd/p274/home/p274-rachelm/Desktop/cr_average.csv")

# Find the 5th to 95th to set range for figure
quantiles_df <- df_wide %>%
  select(-id) %>%
  summarize(across(
    .cols = everything(),
    .fns = list(
      p5 = ~ quantile(.x, 0.05, na.rm = TRUE),
      p95 = ~ quantile(.x, 0.95, na.rm = TRUE)
    )
  ))

max <- max(as.matrix(abs(quantiles_df)), na.rm = TRUE)
print(max)
