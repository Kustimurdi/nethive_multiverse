setwd("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_plotting")
#renv::activate()
renv::activate("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_plotting")

library(ggplot2)
library(readr)

df <- read_csv("data/dt_df.csv")

# stable mapping even if a facet is missing a category
df$kind <- factor(df$kind, levels = c("train", "suppress"))

p <- ggplot(df, aes(x = factor(task_id), y = mean_dt, color = kind)) +
  geom_point(position = position_dodge(width = 0.4), size = 2.8) +
  geom_errorbar(
    aes(ymin = mean_dt - std_dt, ymax = mean_dt + std_dt),
    width = 0.15,
    position = position_dodge(width = 0.4)
  ) +
  facet_wrap(~ agent, ncol = 3, scales = "fixed") +
  labs(x = "task id", y = "mean Δt") +
  theme_bw(base_size = 14) +
  theme(
    legend.title = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave("plots/mean_dt_per_agent.pdf", p, width = 10, height = 6, device = cairo_pdf)
