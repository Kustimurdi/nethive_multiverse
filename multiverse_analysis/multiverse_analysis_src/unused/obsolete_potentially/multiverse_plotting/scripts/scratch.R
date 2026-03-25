setwd("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_plotting")
#renv::activate()
renv::activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_plotting")

library(ggplot2)
library(readr)

preview_plot <- function(p,
                         path = "plots/_preview.png",
                         width = 1400,
                         height = 900,
                         res = 150) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  png(path, width = width, height = height, res = res)
  print(p)
  dev.off()
  message("Updated preview: ", normalizePath(path))
}









# start viewer device once per session
hgd()

df <- read_csv("data/dt_df.csv")
df$kind <- factor(df$kind, levels = c("train", "suppress"))

p <- ggplot(df, aes(x=factor(task_id), y=mean_dt, color=kind)) +
  geom_point(position=position_dodge(0.4), size=2.8) +
  geom_errorbar(aes(ymin=mean_dt-std_dt, ymax=mean_dt+std_dt),
                position=position_dodge(0.4), width=0.15) +
  facet_wrap(~agent, ncol=3) +
  theme_bw(base_size=14) +
  labs(x="task id", y="mean Δt") +
  theme(legend.title=element_blank())

print(p)   # shows in viewer
