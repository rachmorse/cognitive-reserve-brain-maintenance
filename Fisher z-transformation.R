
if (!require("compare")) {install.packages("compare"); require("compare")}


waha_fc <- read_delim("/tsd/p274/data/durable/projects/p027-cr_bm/rsFC/WAHAFiltAndResid.csv", 
                         delim = ";", escape_double = FALSE, trim_ws = TRUE)

cobra_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_Fisher-z-trans_FiltAndResid_Cobra_smallBB.csv")
cobra_no_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_no-Fisher-z-trans_FiltAndResid_Cobra_smallBB.csv")

betula_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_Fisher-z-trans_FiltAndResid_Betula_smallBB.csv")
betula_no_z <- read_csv("/tsd/p274/data/durable/projects/p027-cr_bm/Umu/FC_shirer_CobraBetula/FC_Shirer90x90_no-Fisher-z-trans_FiltAndResid_Betula_smallBB.csv")


# Define the Fisher Z transformation function
fisher_z <- function(r) {
  0.5 * log((1 + r) / (1 - r))
}

# Try the  Fisher Z transformation for Cobra and compare
cobra_no_z <- cobra_no_z %>%
  mutate_at(vars(-"...1"), fisher_z)

cobra_no_z <- cobra_no_z[order(row.names(cobra_no_z)),]
cobra_z <- cobra_z[order(row.names(cobra_z)),]

identical <- compare(cobra_no_z, cobra_z)
print(identical)

# Try for Betula 
betula_no_z <- betula_no_z %>%
  mutate_at(vars(-"...1"), fisher_z)

betula_no_z <- betula_no_z[order(row.names(betula_no_z)),]
betula_z <- betula_z[order(row.names(betula_z)),]

identical <- compare(betula_no_z, betula_z)
print(identical)

# Now transform WAHA data 
waha_fc_z <- waha_fc %>%
  mutate_at(vars(-"code",-"timepoint"), fisher_z)

# Export waha to a CSV file
write.csv(waha_fc_z, "/tsd/p274/data/durable/projects/p027-cr_bm/rsFC/WAHAFiltAndResid_Fisher-z-trans.csv", row.names = FALSE)




