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
  select(id, sn_columns)

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
  group_by(id, ROI, Timepoint) %>%
  summarize(
    avg_connectivity_per_tp = mean(connectivity, na.rm = TRUE),
    .groups = "drop"
  )

# Step 4: Compute the final average across all timepoints, resulting in one value per ROI per participant.
df_roi_final_participant <- df_roi_time_participant %>%
  group_by(id, ROI) %>%
  summarize(
    ROI_av = mean(avg_connectivity_per_tp, na.rm = TRUE),
    .groups = "drop"
  )

sn_average_df <- df_roi_final_participant %>% 
  filter(!is.na(ROI_av))

# Pivot back to wide
df_wide <- sn_average_df %>%
  pivot_wider(
    id_cols      = id,  # Rows identified by participant
    names_from   = ROI,          # Columns become each ROI
    values_from  = ROI_av        # Values are the average connectivity
  )

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
  select(id, reverse_cr, pathway) %>%
  filter(pathway == "CR Pathway")

# Merge
image_data <- merge(cr_data, df_wide, by = c("id"), all = TRUE) # This keeps all participants so need to drop NAs

# Create CR groups
image_data$cr_group <- cut(image_data$reverse_cr,
                                breaks = quantile(image_data$reverse_cr, probs = seq(0, 1, (1 / 2)), na.rm = TRUE),
                                include.lowest = TRUE,
                                labels = c("Low CR", "High CR")
)

# Filter subs missing the needed data
image_data <- image_data %>% 
  filter(!is.na(cr_group) & !is.na(AS_L_ins))

# Create a final df with one average value for each ROI divided by low and high CR group
df_summary <- image_data %>%
  group_by(cr_group) %>%
  summarize(
    across(where(is.numeric), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

write.csv(df_summary, file = "/tsd/p274/home/p274-rachelm/Desktop/cr_average.csv")

