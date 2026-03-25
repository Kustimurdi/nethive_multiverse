# ----------------------------
# scratch_mean_dt.R
# Interactive plotting via a persistent preview PNG
# ----------------------------

# 1) Ensure working directory is the project root -----------------------------
# If you always start R in multiverse_plotting/, you can delete this block.
proj <- Sys.getenv("MULTIVERSE_PLOTTING_ROOT", unset = "")
if (nzchar(proj)) {
  setwd(proj)
}

# Optional: print where we are (helps debugging)
cat("Working directory:", getwd(), "\n")

# 2) Load packages (renv should already be active when started in project dir)-
suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

# 3) Helper: render plot to a preview PNG ------------------------------------
preview_plot <- function(p,
                         path = "plots/_preview.png",
                         width = 1600,
                         height = 1000,
                         res = 150) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  png(path, width = width, height = height, res = res)
  print(p)
  dev.off()
  cat("Updated preview:", normalizePath(path), "\n")
}

# 4) Load data ---------------------------------------------------------------
# Expect columns: agent, task_id, kind, mean_dt, std_dt
csv_path <- "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_plotting/data/mean_dt/mean_dt_2b2t15c10000e10+dt90+int_param_85_rep_6.CSV"
df <- read_csv(csv_path, show_col_types = FALSE)

# Stable mapping even if one kind is missing in some facets
df$kind <- factor(df$kind, levels = c("train", "suppress"))

# Optional: drop rows with missing mean_dt or std_dt
df <- df[!is.na(df$mean_dt) & !is.na(df$std_dt), ]

# 5) Build plot (edit this section while iterating) --------------------------
p <- ggplot(df, aes(x = factor(task_id),
                    y = mean_dt,
                    color = kind,
                    group = kind)) +
  geom_point(position = position_dodge(width = 0.4), size = 2.8) +
  geom_errorbar(
    aes(ymin = mean_dt - std_dt, ymax = mean_dt + std_dt),
    position = position_dodge(width = 0.4),
    width = 0.15
  ) +
  facet_wrap(~ agent, ncol = 3, scales = "fixed") +
  labs(x = "task id", y = "mean Δt") +
  theme_bw(base_size = 14) +
  theme(
    legend.title = element_blank(),
    panel.grid.minor = element_blank()
  )

df$agent   <- factor(df$agent, levels = sort(unique(df$agent)))
df$task_id <- factor(df$task_id, levels = sort(unique(df$task_id)))
df$kind    <- factor(df$kind, levels = c("train", "suppress"))

p <- ggplot(df, aes(x = task_id, y = mean_dt, color = kind)) +
  geom_point(position = position_dodge(width = 0.4), size = 2.8) +
  geom_errorbar(
    aes(ymin = mean_dt - std_dt, ymax = mean_dt + std_dt),
    width = 0.15,
    position = position_dodge(width = 0.4)
  ) +
  facet_wrap(~ agent, ncol = 3, labeller = label_both) +
  labs(x = "task id", y = "mean Δt") +
  theme_bw(base_size = 16) +
  theme(
    strip.text = element_text(size = 14),
    legend.title = element_blank(), legend.position = "bottom",
    panel.grid.minor = element_blank()
  )


# 6) Preview (run this repeatedly) -------------------------------------------
preview_plot(p)

# 7) Final save (ONLY run once you're happy) ---------------------------------
# Uncomment when ready.
folder_path <- "/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/thesis_plots/mean_dt"

name <- basename(csv_path)
name <- paste(folder_path, "mean_dt_per_agent.pdf", sep = "/")
ggsave(name, p, width = 10, height = 6, device = cairo_pdf)
