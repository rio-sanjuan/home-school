library(tidyverse)
library(car)

set.seed(0)

X1 <- rnorm(100)
X2 <- 2 * X1 + rnorm(100) # X2 is highly correlated with X1
X3 <- -1 * X1 + rnorm(100) # X3 is highly correlated with X1 and X2

df <- data.frame(X1, X2, X3)

max_vif <- 5.0

while (TRUE) {
    vif_values <- vif(lm(df))
    max_vif_feature <- names(which.max(vif_values))
    max_vif_value <- max(vif_values)

    if (max_vif_value <= max_vif) break

    df <- df %>%
        select(-max_vif_feature)
    print(paste("Removed", max_vif_feature, "due to high VIF value:", max_vif_value, "\n"))
}

print("Remaining Features:")
print(names(df))
