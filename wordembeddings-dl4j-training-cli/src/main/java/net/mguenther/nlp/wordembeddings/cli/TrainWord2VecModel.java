package net.mguenther.nlp.wordembeddings.cli;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;

import java.io.File;
import java.lang.invoke.MethodHandles;
import java.util.concurrent.Callable;

@CommandLine.Command(
        name = "train-word2vec-model",
        description = "Trains a word2vec model using Deeplearning4J based on a sentence-by-line textfile."
)
public class TrainWord2VecModel implements Callable<Integer> {

    private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    @CommandLine.Parameters(
            index = "0",
            description = "The training corpus for the word2vec model")
    private File inputFilename;

    @CommandLine.Option(
            names = {"-o", "--output"},
            description = "Sets the output filename")
    private File outputFilename;

    @CommandLine.Option(
            names = {"-f", "--force"},
            defaultValue = "false",
            description = "Override an already existing output file")
    private boolean overrideOutputIfExists;

    @CommandLine.Option(
            names = {"--minWordFrequency"},
            defaultValue = "5",
            description = "Sets the minimal element frequency for elements found in the training corpus. All elements below this threshold will be removed before training.")
    private Integer minWordFrequency;

    @CommandLine.Option(
            names = {"--iterations"},
            defaultValue = "1",
            description = "Sets how many iterations should be done over batched sequences.")
    private Integer iterations;

    @CommandLine.Option(
            names = {"--layerSize"},
            defaultValue = "100",
            description = "Sets the number of dimensions for outcome vectors")
    private Integer layerSize;

    @CommandLine.Option(
            names = {"--seed"},
            defaultValue = "42",
            description = "Sets the seed value for the internal random number generator")
    private Integer seed;

    @CommandLine.Option(
            names = {"--windowSize"},
            defaultValue = "5",
            description = "Sets the window size for Skip-Gram training")
    private Integer windowSize;

    @CommandLine.Option(
            names = {"-v", "--verbose"},
            defaultValue = "false",
            description = "Increases the amount of log output")
    private boolean verbose;

    @Override
    public Integer call() throws Exception {

        if (outputFilename == null) {
            final String s = inputFilename.toPath().getFileName().toString();
            final String t = s.substring(0, s.lastIndexOf(".")) + ".bin";
            log.info("No output filename has been provided. Using '{}'.", t);
            outputFilename = new File(t);
        }

        if (!isReadable(inputFilename)) {
            log.error("The source file '{}' does not exist or is not readable.", inputFilename.toString());
            return 1;
        }

        if (exists(outputFilename) && !overrideOutputIfExists) {
            log.error("Unable to write to output file '{}'.", outputFilename.toString());
            return 1;
        }

        int returnCode = 0;
        try {
            trainModel(inputFilename, outputFilename);
        } catch (Exception e) {
            if (verbose) {
                log.error("An error occured while attempting to train the model.", e);
            } else {
                log.error("An error occured while attempting to train the model: {}", e.getMessage());
            }
            returnCode = 1;
        }

        return returnCode;
    }

    private void showBanner() {
        // this is required to also show the banner on normal execution
        final String[] banner = new CommandLine(new TrainWord2VecModel())
                .getCommandSpec()
                .usageMessage()
                .header();
        for (String line : banner) {
            log.info(CommandLine.Help.Ansi.AUTO.string(line));
        }
    }

    private boolean isReadable(final File f) {
        return f.exists() && f.canRead();
    }

    private boolean exists(final File f) {
        return f.exists();
    }

    private void trainModel(final File corpusLocation, final File trainedModelLocation) throws Exception {
        final SentenceIterator iter = new LineSentenceIterator(corpusLocation);
        iter.setPreProcessor(new SentencePreProcessor() {
            int i = 0;

            @Override
            public String preProcess(String sentence) {
                i++;
                if (verbose && i % 100_000 == 0) log.info("Processed '{}' sentences.", i);
                return sentence.toLowerCase();
            }
        });

        final TokenizerFactory t = new DefaultTokenizerFactory();
        // CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
        // So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
        // Additionally, it forces lower case for all tokens.
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model ...");
        final Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(minWordFrequency)
                .iterations(iterations)
                .layerSize(layerSize)
                .windowSize(windowSize)
                .seed(seed)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting word2vec model ...");
        vec.fit();

        log.info("Writing word vectors to file ...");
        WordVectorSerializer.writeWord2VecModel(vec, trainedModelLocation);
    }

    public static void main(String[] args) {
        final int exitCode = new CommandLine(new TrainWord2VecModel()).execute(args);
        System.exit(exitCode);
    }
}
