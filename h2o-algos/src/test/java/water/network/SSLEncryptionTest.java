package water.network;

import hex.Distribution;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import org.junit.Assert;
import water.TestUtil;
import water.fvec.Frame;
import water.util.MathUtils;

import static org.junit.Assert.assertEquals;

/**
 * This class is used to capture TCP packets while training a model
 * The result is then used to check if SSL encryption is working properly
 */
public class SSLEncryptionTest extends TestUtil {

    public static void main(String[] args) {
        if(args.length > 0 && !args[0].isEmpty()) {
            testGBMRegressionGaussianSSL(args[0]);
        } else {
            testGBMRegressionGaussianNonSSL();
        }
        System.exit(0);
    }

    public static void testGBMRegressionGaussianNonSSL() {
        stall_till_cloudsize(3);
        testGBMRegressionGaussian();
    }

    public static void testGBMRegressionGaussianSSL(String prop) {
        stall_till_cloudsize(new String[] {"-ssl_config", prop}, 3);
        testGBMRegressionGaussian();
    }

    private static void testGBMRegressionGaussian() {
        GBMModel gbm = null;
        Frame fr = null, fr2 = null;
        try {
            fr = parse_test_file("./smalldata/gbm_test/Mfgdata_gaussian_GBM_testing.csv");
            GBMModel.GBMParameters parms = new GBMModel.GBMParameters();
            parms._train = fr._key;
            parms._distribution = Distribution.Family.gaussian;
            parms._response_column = fr._names[1]; // Row in col 0, dependent in col 1, predictor in col 2
            parms._ntrees = 1;
            parms._max_depth = 1;
            parms._min_rows = 1;
            parms._nbins = 20;
            // Drop ColV2 0 (row), keep 1 (response), keep col 2 (only predictor), drop remaining cols
            String[] xcols = parms._ignored_columns = new String[fr.numCols()-2];
            xcols[0] = fr._names[0];
            System.arraycopy(fr._names,3,xcols,1,fr.numCols()-3);
            parms._learn_rate = 1.0f;
            parms._score_each_iteration=true;

            GBM job = new GBM(parms);
            gbm = job.trainModel().get();
            Assert.assertTrue(job.isStopped()); //HEX-1817

            // Done building model; produce a score column with predictions
            fr2 = gbm.score(fr);
            //job.response() can be used in place of fr.vecs()[1] but it has been rebalanced
            double sq_err = new MathUtils.SquareError().doAll(fr.vecs()[1],fr2.vecs()[0])._sum;
            double mse = sq_err/fr2.numRows();
            assertEquals(79152.12337641386,mse,0.1);
            assertEquals(79152.12337641386,gbm._output._scored_train[1]._mse,0.1);
            assertEquals(79152.12337641386,gbm._output._scored_train[1]._mean_residual_deviance,0.1);
        } finally {
            if( fr  != null ) fr .remove();
            if( fr2 != null ) fr2.remove();
            if( gbm != null ) gbm.remove();
        }
    }

}
