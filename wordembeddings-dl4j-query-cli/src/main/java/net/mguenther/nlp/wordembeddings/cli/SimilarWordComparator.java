package net.mguenther.nlp.wordembeddings.cli;

import java.util.Comparator;

public class SimilarWordComparator implements Comparator<Similar> {

    @Override
    public int compare(final Similar left, final Similar right) {
        if (left.getSimilarity() == right.getSimilarity()) {
            return 0;
        } else {
            return (int) Math.signum(right.getSimilarity() - left.getSimilarity());
        }
    }
}
