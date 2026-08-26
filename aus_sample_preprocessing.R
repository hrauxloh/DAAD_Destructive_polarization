## ---------------------------------------------------------------
## R data pre-processing: australia_498sample_climatechange
## ---------------------------------------------------------------

library(dplyr)
library(stringr)
library(tidyr)

## 1. New dataset with only document_id, full_text -----------------
df <- australia_498sample_climatechange %>%
  select(document_id, full_text)

## 2. Remove nonword entities (\n) ----------------------------------
df <- df %>%
  mutate(full_text = str_replace_all(full_text, "\\n", " ")) %>%
  mutate(full_text = str_squish(full_text))  # collapse any resulting extra whitespace

## 3. Replace decimals with , ---------------------------------------
## Protects numbers like "3.14" -> "3,14" so the fullstop-split below
## doesn't treat them as sentence boundaries.
df <- df %>%
  mutate(full_text = str_replace_all(full_text, "(\\d)\\.(\\d)", "\\1,\\2"))

## 4. Split by fullstops ---------------------------------------------
## Produces one row per sentence, keeping document_id for traceability.
df_sentences <- df %>%
  mutate(full_text = str_split(full_text, "\\.")) %>%
  unnest(full_text) %>%
  mutate(full_text = str_trim(full_text)) %>%
  filter(full_text != "")  # drop empty fragments left by trailing fullstops

## 5. Add sentence IDs ------------------------------------------------
## sentence_number: position of the sentence within its own document (resets per document)
## sentence_id: unique ID across the whole dataset, e.g. "doc1_s1"
df_sentences <- df_sentences %>%
  group_by(document_id) %>%
  mutate(sentence_number = row_number()) %>%
  ungroup() %>%
  mutate(sentence_id = paste0(document_id, "_s", sentence_number)) %>%
  relocate(document_id, sentence_id, sentence_number, full_text)
df_sentences <- df_sentences %>%
  filter(str_length(full_text) >= 5)
