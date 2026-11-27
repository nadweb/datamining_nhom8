# -*- coding: utf-8 -*-
"""[STEP 4] Zero-Rule Majority Class Baseline."""
from sklearn.dummy import DummyClassifier


def build_majority_classifier():
    return DummyClassifier(strategy="most_frequent")
