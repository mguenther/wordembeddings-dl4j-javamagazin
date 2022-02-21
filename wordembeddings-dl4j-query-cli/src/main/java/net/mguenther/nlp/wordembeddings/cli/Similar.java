package net.mguenther.nlp.wordembeddings.cli;

import java.text.NumberFormat;

public class Similar {

    private final String word;

    private final String similarTo;

    private final double similarity;

    public Similar(final String word, final String similarTo, final double similarity) {
        this.word = word;
        this.similarTo = similarTo;
        this.similarity = similarity;
    }

    public String getWord() {
        return word;
    }

    public String getSimilarTo() {
        return similarTo;
    }

    public double getSimilarity() {
        return similarity;
    }

    @Override
    public String toString() {
        return String.format("%s (%s)", similarTo, NumberFormat.getPercentInstance().format(similarity));
    }
}
