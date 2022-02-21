package net.mguenther.nlp.wordembeddings.cli;

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

public class LowercasePreProcessor implements SentencePreProcessor {

    @Override
    public String preProcess(final String sentence) {
        return sentence.trim().toLowerCase();
    }
}
