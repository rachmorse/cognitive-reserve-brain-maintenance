# Install and load packages
if (!require("dplyr")) {
  install.packages("dplyr")
  require("dplyr")
}
if (!require("compare")) {
  install.packages("compare")
  require("compare")
}
if (!require("readr")) {
  install.packages("readr")
  require("readr")
}

# Read in data that needs to be z-transformed
waha_fc <- read_delim("/tsd/p274/data/durable/projects/p027-cr_bm/rsFC/WAHAFiltAndResid.csv",
  delim = ";", escape_double = FALSE, trim_ws = TRUE
)

# Read in additional pre z-transformed data and post z-transformed data for another cohort
cobra_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_Fisher-z-trans_FiltAndResid_Cobra_smallBB.csv")
cobra_no_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_no-Fisher-z-trans_FiltAndResid_Cobra_smallBB.csv")

# Define the Fisher z-transformation function
fisher_z <- function(r) {
  0.5 * log((1 + r) / (1 - r))
}

# Try the  Fisher z-transformation for data that has been z-transformed to compare output of this function with already transformed data
cobra_no_z <- cobra_no_z %>%
  mutate_at(vars(-"...1"), fisher_z)

cobra_no_z <- cobra_no_z[order(row.names(cobra_no_z)), ]
cobra_z <- cobra_z[order(row.names(cobra_z)), ]

# Check that the output is the same
identical <- compare(cobra_no_z, cobra_z)
print(identical)

# Now z-transform WAHA data
waha_fc_z <- waha_fc %>%
  mutate_at(vars(-"code", -"timepoint"), fisher_z)

# Export WAHA z-transformed data to a CSV file
write.csv(waha_fc_z, "/tsd/p274/data/durable/projects/p027-cr_bm/rsFC/WAHAFiltAndResid_Fisher-z-trans.csv", row.names = FALSE)
