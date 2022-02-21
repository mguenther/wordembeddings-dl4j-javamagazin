package net.mguenther.nlp.wordembeddings.cli;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import picocli.CommandLine;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@CommandLine.Command(
        name = "query-word2vec-model",
        description = "Loads a word2vec model and provides a simple query interface."
)
public class QueryWord2VecModel implements Callable<Integer> {

    private static final String PROMPT = "query> ";

    @CommandLine.Parameters(
            index = "0",
            description = "The binary file that contains the word2vec model")
    private File inputFile;

    private boolean running = true;

    private Word2Vec model;

    @Override
    public Integer call() throws Exception {

        if (!isReadable(inputFile)) {
            System.err.printf("The word2vec model at '%s' does not exist or is not readable.%n", inputFile.toString());
            return 1;
        }

        try {
            loadModel();
        } catch (Exception e) {
            System.err.printf("Unable to load word2vec model from file '%s'.%n", inputFile.toString());
            return 1;
        }

        while (running) {
            final var in = readLine(PROMPT);

            if ("/exit".equalsIgnoreCase(in)) {
                running = false;
            } else {
                var tokens = in.split(" ");
                if (tokens.length > 2) {
                    System.out.println("Unrecognized command.");
                }
                if (tokens.length == 1) {
                    var word = tokens[0].trim();
                    if (!exists(word)) {
                        System.out.printf("The word '%s' is not known to the word2vec model.%n", word);
                    } else {
                        nearest(word, 5)
                                .stream()
                                .sorted(new SimilarWordComparator())
                                .forEach(w -> System.out.println(StringUtils.EMPTY + w));
                    }
                }
                if (tokens.length == 2) {
                    var word = tokens[0].trim();
                    var otherWord = tokens[1].trim();
                    if (!exists(word)) {
                        System.out.printf("The word '%s' is not known to the word2vec model.%n", word);
                    } else if (!exists(otherWord)) {
                        System.out.printf("The word '%s' is not known to the word2vec model.%n", otherWord);
                    } else {
                        var similarity = similarity(word, otherWord);
                        System.out.println(StringUtils.EMPTY + similarity);
                    }
                }
            }
        }

        return 0;
    }

    private boolean isReadable(final File f) {
        return f != null && f.exists() && f.canRead();
    }

    private void loadModel() {
        System.out.printf("Attempting to load word2vec model from file '%s'.%n", inputFile.toString());
        final long start = System.nanoTime();
        model = WordVectorSerializer.readWord2VecModel(inputFile);
        final long end = System.nanoTime();
        final long duration = TimeUnit.NANOSECONDS.toMillis(end - start);
        System.out.printf("Successfully loaded word2vec model from file '%s'. Took %s ms.%n", inputFile.toString(), duration);
        System.out.println(info());
    }

    private String info() {
        return "Parameterization:" + "\n" +
                "  layer size            : " + model.getLayerSize() + "\n" +
                "  window size           : " + model.getWindow() + "\n" +
                "  minimum word frequency: " + model.getMinWordFrequency() + "\n";
    }

    private boolean exists(final String word) {
        return model.hasWord(word);
    }

    private Collection<Similar> nearest(final String word, final int howMany) {
        return model
                .wordsNearest(word, howMany)
                .stream()
                .map(w -> new Similar(word, w, similarity(word, w)))
                .sorted((left, right) -> {
                    if (left.getSimilarity() == right.getSimilarity()) {
                        return 0;
                    } else {
                        return (int) Math.signum(right.getSimilarity() - left.getSimilarity());
                    }
                })
                .collect(Collectors.toList());
    }

    private double similarity(final String word, final String otherWord) {
        return model.similarity(word, otherWord);
    }

    private String readLine(String format, Object... args) throws IOException {
        if (System.console() != null) {
            return System.console().readLine(format, args);
        }
        System.out.printf(format, args);
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        return reader.readLine();
    }

    public static void main(String[] args) {
        final int exitCode = new CommandLine(new QueryWord2VecModel()).execute(args);
        System.exit(exitCode);
    }
}
