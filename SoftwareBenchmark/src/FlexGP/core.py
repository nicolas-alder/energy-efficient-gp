
from .load_libraries import *

from . import count_enabler
from . import config
from . import cg_tracker

from .utils import set_global_section, get_global_section, get_dataset, reset_wrapped_functions, average_testpoint_section
from .utils import converter2mpf, log, exponential, power_two, len, range, lt, le, eq, ne, ge, gt, add_int, sub_int, mul_int, floordiv_int, absolute
from .utils import mul_float, add_float, sub_float, truediv_float, sqrt_float
from .utils import np_add, np_divide, np_dot, np_subtract, np_multiply, np_sum, np_inner, np_sign, np_max
from .utils import pretty_print_sections_info_to_csv, counts_info_to_csv, pretty_print_sections_info

##############################################################################
# Purpose: Create a covariance matrix with a rbf kernel
# Source: Structure Scikit Learn, then adapted to MP by myself and selected functions with ChatGPT idea sourcing
# and subsequent modification.
##############################################################################


def pdist(X):
    n, m = np.shape(X)
    Y = np.full(floordiv_int(mul_int(n, (sub_int(n, 1))), 2), gmpy2.mpfr("0.0"))
    k = 0
    for i in range(0, n):
        for j in range(add_int(i,1), n):
            d = np_sum(power_two(np_subtract(X[i], X[j])))
            Y[k] = d
            k = add_int(k, 1)
    return Y


def compute_distance(X, Y, p, i):
    distances = []
    for j in range(p):
        d = np_sum(power_two(np_subtract(X[i], Y[j])))
        distances.append(d)
    return distances

def sequential_pairwise_distances(X, Y=None):
    n, m = np.shape(X)
    if Y is None:
        Y = X.copy()
    p, _ = np.shape(Y)
    Y_result = np.full((mul_int(n, p)), gmpy2.mpfr("0.0"))
    k = 0
    for i in tqdm(range(n), desc="Calculating distances"):
        distances = compute_distance(X, Y, p, i)
        for d in distances:
            Y_result[k] = d
            k = add_int(k,1)
    return Y_result


def rbf_kernel(X_train, length_scale, noise, output_scale, fit=True, Y=None):

    if fit:
        dists = sequential_pairwise_distances(np_divide(X_train, length_scale))
        len_X = len(X_train)
        K = np_multiply(output_scale, exponential(np_multiply(dists, gmpy2.mpfr("-0.5")))).reshape(len_X,len_X)
        np.fill_diagonal(K, 1)
        K = np_add(K , np_multiply(converter2mpf(np.eye(len(X_train))), noise))
        print("Finished RBF")
        return K
    else:
        dists = sequential_pairwise_distances(np_divide(X_train, length_scale), np_divide(Y, length_scale))
        K = np_multiply(output_scale , exponential(np_multiply(dists, gmpy2.mpfr("-0.5"))))
        K = K.reshape(-1, len(Y))
        return K


def find_max(lst):
    if not lst:
        raise ValueError("The list cannot be empty")
    maximum = lst[0]
    for number in lst:
        if gt(number, maximum):
            maximum = number
    return lst.index(maximum)


def pivoted_chol(K, M):
    print("start pivoted cholesky preconditioner")
    err_tol = config.experiment.config["matrix_inversion"]["preconditioner_rank_tolerance"]
    get_diag = lambda: np.diag(K).copy()
    get_row = lambda i: K[i, :]
    d = np.copy(get_diag())
    N = len(d)
    pi = list(range(N)) # To-Do: Range Operation bewerten

    R = np.full((M, N), gmpy2.mpfr("0.0"))
    err = np_sum(absolute(d))

    m = 0
    while (lt(m,M) and gt(err,err_tol)):
        i = add_int(m, find_max([d[pi[j]] for j in range(m, N)]))
        tmp = pi[m]
        pi[m] = pi[i]
        pi[i] = tmp
        R[m, pi[m]] = sqrt_float(d[pi[m]])
        Apim = get_row(pi[m])
        for i in range(add_int(m, 1), N):
            if gt(m,0):
                ip = np_inner(R[:m, pi[m]], R[:m, pi[i]])
            else:
                ip = gmpy2.mpfr("0.0")
            R[m, pi[i]] = truediv_float(sub_float(Apim[pi[i]], ip), R[m, pi[m]])
            d[pi[i]] = sub_float(d[pi[i]], *power_two(np.array([R[m, pi[i]]])))
        err = np_sum(np.array([d[pi[i]] for i in range(add_int(m, 1), N)]))
        m = add_int(m,1)
    R = R[:m, :]
    print("end pivoted cholesky preconditioner")
    return (R.T)


# Source: http://www.quantstart.com/articles/Cholesky-Decomposition-in-Python-and-NumPy/
def cholesky(A):
    n = len(A)
    L = [[gmpy2.mpfr("0.0")] * n for i in range(n)]
    for i in tqdm(range(n), desc="Calculating cholesky"):
        for k in range(add_int(i, 1)):
            tmp_sum = np_sum(np.array([mul_float(L[i][j], L[k][j]) for j in range(k)]))
            if eq(i, k):
                L[i][k] = sqrt_float(sub_float(A[i][i], tmp_sum))
            else:
                L[i][k] = (mul_float(truediv_float(gmpy2.mpfr("1.0"), L[k][k]) , sub_float(A[i][k] , tmp_sum)))
    return np.array(L)


# Source: ChatGPT
def transpose(matrix):
    count_enabler.counting_enabled.old_state = copy.copy(count_enabler.counting_enabled.condition) # because len command which is part of reshaping (transpose)
    count_enabler.counting_enabled.condition = False
    result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state
    return result


def forward_substitution(L, y):
    print("start forward_substitution")
    n = len(y)
    x = np.full(n, gmpy2.mpfr("0.0"))
    x[0] = np_divide(np_subtract(y[0], gmpy2.mpfr("0.0")), L[0, 0])
    for i in range(1,n):
        x[i] = np_divide(np_subtract(y[i], np_dot(L[i, :i].reshape(1,-1), x[:i].reshape(-1,1)).flatten().tolist()) , L[i, i])[0]
    print("end forward_substitution")
    return x


def backward_substitution(L, z):
    print("start backward_substitution")
    n = len(z)
    x = np.full(n, gmpy2.mpfr("0.0"))
    init = sub_int(n,1)
    x[init] = np_divide(np_subtract(z[init], gmpy2.mpfr("0.0")), L[init, init])
    for i in range(sub_int(n,2), -1, -1):
        i_plus_1 = add_int(i,1)
        x[i] = np_divide(np_subtract(z[i], np_dot(L[i, i_plus_1:].reshape(1,-1), x[i_plus_1:].reshape(-1,1)).flatten().tolist()) , L[i, i])[0]
    print("end backward_substitution")
    return x


def cholesky_inverse(B):
    L = cholesky(B)
    n, _ = np.shape(B)
    I = converter2mpf(np.eye(n))
    B_inv = []
    for i in range(n):
        z = forward_substitution(L, I[:,i])
        inv_column = backward_substitution(L.T, z)
        B_inv.append(inv_column)
    return np.array(B_inv)


# Source = https://stackoverflow.com/questions/53564529/woodbury-identity-for-fast-matrix-inversion-slower-than-expected
def efficient_woodbury(A, U, V, k):
    A_inv_diag = np_divide(np.full_like(np.diag(A), gmpy2.mpfr("1.")), np.diag(A))  # note! A_inv_diag is a vector!
    B = np_add(converter2mpf(np.eye(k)), np_dot(np_multiply(V, A_inv_diag), U))
    B_inv = cholesky_inverse(B)
    A_inv_diag_square = np.full_like(A, gmpy2.mpfr("0.0"))
    np.fill_diagonal(A_inv_diag_square, A_inv_diag)
    return np_subtract(A_inv_diag_square, np_dot(np_dot(np_multiply(U.T, np_multiply(A_inv_diag, A_inv_diag).reshape(-1,A_inv_diag.shape[0])).T, B_inv) ,V))


def logsumtransform(rk, zk):
    sign = np_multiply(np_sign(rk), np_sign(zk))
    y = np_add(log(absolute(rk)) , log(absolute(zk)))
    y_max = np_max(y)
    res = add_float(*log([np_sum(np_multiply(sign , exponential(np_subtract(y, y_max))))]), y_max)
    return res


def frobenius_norm(vector):
    sqrt_float(np_sum(power_two([absolute(element) for element in vector.flatten().tolist()])))
    return sqrt_float(np_sum(power_two([absolute(element) for element in vector.flatten().tolist()])))


def ModifiedCG(K, y, tol, iterations=1000, reorthogonalization=True, preconditioner=None, use_p_cholesky_preconditioner=False):
    k = 0
    xk = np.full_like(y, gmpy2.mpfr("0"))
    len_y_int = len(y)
    len_y =  gmpy2.mpfr(len_y_int)
    y_scaled = np_divide(y, len_y)
    rk = np_subtract(np_dot(K, np_divide(xk, len_y)), y_scaled)  # Matrix vector product K*x=y

    if use_p_cholesky_preconditioner:
        zk = np_dot(preconditioner, rk)
        yk = logsumtransform(rk, zk)

    else:
        yk = logsumtransform(rk, rk)

    u = np.full((1, len(y), 1), gmpy2.mpfr(0.0))
    rk_norm = frobenius_norm(rk)

    with tqdm(bar_format=" Elapsed: {elapsed} , {rate_fmt}  {postfix}",
              postfix=f'Residual = {rk_norm:.4f} | Iterations= {k}'.format(rk_norm, k), ) as t:
        while (gt(rk_norm, tol) and lt(k, iterations)) or not gt(k,0):
            if use_p_cholesky_preconditioner:
                dk = np_multiply(zk, gmpy2.mpfr("-1"))
            else:
                dk = np_multiply(rk, gmpy2.mpfr("-1"))

            cg_section = get_global_section()

            if cg_section == "section_conjugate_gradient_mean":
                set_global_section("section_cg_mvm_only_mean")
            else:
                set_global_section("section_cg_mvm_only_covariance")

            Kdk = np_dot(K, np_divide(dk, len_y).reshape(-1,1))
            set_global_section(cg_section)
            ydk = logsumtransform(dk, Kdk)
            ak = exponential(np_subtract(yk, ydk))  # step size of cg algorithm along the path xk + ak*pk, difference to formula in davies is preconditioner application
            xk = np_add(xk, np_multiply(dk, ak))

            rk = np_add(rk, np_multiply(Kdk, ak))

            if reorthogonalization:
                for j in range(0, k):
                    rk = np_subtract(rk, np_multiply(u[j], np_dot(u[j].T, rk.reshape(-1,1))))

            # reorthogonalization: funktioniert weil u.T*u = 0, siehe alexander davies formel (37)

            rk_norm = frobenius_norm(rk)

            cg_tracker.tracker.saver.append([get_global_section(), k, rk_norm, tol])

            if lt(rk_norm, tol) or eq(add_int(k,1), iterations):
                return xk

            if use_p_cholesky_preconditioner:
                zk = np_dot(preconditioner, rk)
                yknext = logsumtransform(rk, zk)
            else:
                yknext = logsumtransform(rk, rk)

            bk = exponential(np_subtract(yknext, yk))  # ensures conjugancy between search directions, difference to formula in davies is preconditioner application
            yk = yknext
            dk = np_subtract(np_multiply(dk, bk), rk)
            u = np.append(u, dk).reshape(-1, len_y_int, 1)  # save orthonormal vector for later iterations of reorthogonalization
            k = add_int(k, 1)

            t.update()
            t.postfix = f'Residual = {rk_norm:.4f} , Iterations= {k}'.format(rk_norm, k)

def l_bfgs(obj_func, initial_theta, bounds):
    theta_opt, func_min, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=20)
    return theta_opt, func_min
class GaussianProcessRegressor():

    def __init__(self, configuration):
        config.experiment.set_experiment(configuration)
        set_global_section("other")

        count_enabler.counting_enabled.old_state = copy.copy(count_enabler.counting_enabled.condition)
        count_enabler.counting_enabled.condition = False


        self.X_train = get_dataset(datasetName=config.experiment.config["dataset_train"], features=True)
        self.y_train = get_dataset(datasetName=config.experiment.config["dataset_train"], features=False)
        self._y_train_mean = 0
        self._y_train_std = 1
        self.X_test = get_dataset(datasetName=config.experiment.config["dataset_test"], features=True)
        self.y_test = get_dataset(datasetName=config.experiment.config["dataset_test"], features=False)
        self._y_test_mean = 0
        self._y_test_std = 1
        self.noise = config.experiment.config["kernel"]["noise"]
        self.output_scale = config.experiment.config["kernel"]["output_scale"]
        self.length_scales = config.experiment.config["kernel"]["length_scales"]
        self.use_scikit_hyperparameter = config.experiment.config["kernel"]["use_scikit_hyperparameter"]

        # Benchmark first Scikit Learn Baseline to obtain kernel hyperparameter in case we want to use them
        if self.use_scikit_hyperparameter:
            kernel = RBF(1.0)
            self.gpr = GPR(n_restarts_optimizer=20,optimizer= l_bfgs, kernel=kernel, random_state=0, alpha=float(self.noise), normalize_y=False).fit(self.X_train.astype(np.double), self.y_train.astype(np.double))
            rbf_length_scale = gmpy2.mpfr(self.gpr.kernel_.get_params()["length_scale"])
            self.length_scales = rbf_length_scale
        else:
            kernel = RBF(float(self.length_scales[0]), length_scale_bounds="fixed")
            self.gpr = GPR(optimizer=None, kernel=kernel, random_state=0, alpha=float(self.noise), normalize_y=False).fit(self.X_train.astype(np.double), self.y_train.astype(np.double))

        self.scikit_mean_test, self.scikit_std_test = self.gpr.predict(self.X_test.astype(np.double), return_std=True)
        self.scikit_mean_train, self.scikit_std_train = self.gpr.predict(self.X_train.astype(np.double), return_std=True)

        self.rms_scikit_test = mean_squared_error(self.scikit_mean_test, self.y_test.astype(np.double), squared=False)
        self.rms_scikit_train = mean_squared_error(self.scikit_mean_train, self.y_train.astype(np.double), squared=False)

        count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state

        self.X_train_len = len(self.X_train)
        print("\nStart RBF")
        set_global_section("section_rbf_kernel_XX")
        print("Length Scale")
        print(self.length_scales)
        print("Output Scale")
        print(self.output_scale)
        self.K = rbf_kernel(self.X_train, self.length_scales, self.noise, self.output_scale, fit=True)

        set_global_section("other")



    def fit(self):
        set_global_section("other")
        self.cg_tol = config.experiment.config["matrix_inversion"]["cg_tolerance"]
        self.p_chol_rank = config.experiment.config["matrix_inversion"]["preconditioner_rank"]
        self.use_p_cholesky_preconditioner = config.experiment.config["matrix_inversion"]["cholesky_preconditioner"]
        self.cg_iterations_mean = config.experiment.config["matrix_inversion"]["cg_iterations_mean"]
        self.cg_iterations_covariance = config.experiment.config["matrix_inversion"]["cg_iterations_covariance"]
        self.cholesky = config.experiment.config["matrix_inversion"]["cholesky"]
        self.direct_inverse_cg = config.experiment.config["matrix_inversion"]["direct_inverse_cg"]
        self.reorthogonalization = config.experiment.config["matrix_inversion"]["cg_reorthogonalization"]
        self.vanilla_cg = config.experiment.config["matrix_inversion"]["vanilla_cg"]

        if self.use_p_cholesky_preconditioner:
            set_global_section("section_pivoted_cholesky")
            self.Lk = pivoted_chol(self.K, self.p_chol_rank)
            set_global_section("section_woodbury")
            self.preconditioner = efficient_woodbury(A=self.K, U=self.Lk, V=self.Lk.T, k=self.p_chol_rank)
            set_global_section("section_conjugate_gradient_mean")
            self.alpha = ModifiedCG(self.K, self.y_train, iterations = self.cg_iterations_mean, tol=self.cg_tol, use_p_cholesky_preconditioner=True,
                                    preconditioner=self.preconditioner, reorthogonalization = self.reorthogonalization)
            set_global_section("other")

        elif self.cholesky and not self.direct_inverse_cg:
            print("start cholesky decomposition")
            set_global_section("section_cholesky_mean")
            self.L = cholesky(self.K)
            self.z = forward_substitution(self.L, self.y_train)
            self.alpha = backward_substitution(self.L.T, self.z).reshape(-1,1)
            set_global_section("other")

        elif self.direct_inverse_cg:
            set_global_section("section_conjugate_gradient_mean")
            K_inv = []
            for column in converter2mpf(np.eye(self.X_train_len).T):
                k_col_inv = ModifiedCG(self.K, column.reshape(-1,1), iterations=self.cg_iterations_mean, tol=self.cg_tol, reorthogonalization = self.reorthogonalization)
                K_inv.append(k_col_inv)
            self.K_inv = np.array(K_inv).reshape(self.X_train_len, -1)
            self.alpha = np_dot(self.K_inv, self.y_train)
            set_global_section("other")

        elif self.vanilla_cg: # conjugate gradient
            set_global_section("section_conjugate_gradient_mean")
            self.alpha = ModifiedCG(self.K, self.y_train, iterations=self.cg_iterations_mean, tol=self.cg_tol, reorthogonalization = self.reorthogonalization)
            set_global_section("other")

        else:
            raise Exception("Sorry, incorrect configuration file")

        return self

    def predict(self, testset = True):
        set_global_section("other_predict")
        if testset:
            test = self.X_test
            self.y_cov_test = []
            self.y_mean_test = []
        else:
            count_enabler.counting_enabled.old_state = copy.copy(
                count_enabler.counting_enabled.condition)  # because len command which is part of reshaping (transpose)
            count_enabler.counting_enabled.condition = False
            test = self.X_train
            self.y_cov_train = []
            self.y_mean_train = []


        for testpoint_x in test:
            testpoint_x = testpoint_x.reshape(1,-1)
            print("Start RBF")
            set_global_section("section_rbf_kernel_X*X")
            K_test_train = rbf_kernel(testpoint_x, self.length_scales, self.noise, self.output_scale, fit=False, Y=self.X_train)
            set_global_section("other_predict")
            if testset:
                self.y_mean_test.append(np_dot(K_test_train, self.alpha)[0][0])
            else:
                self.y_mean_train.append(np_dot(K_test_train, self.alpha)[0][0])
            append_later = []
            set_global_section("section_rbf_kernel_XX*")

            for prediction_counter, vector in enumerate(K_test_train):
                if self.use_p_cholesky_preconditioner:
                    set_global_section("section_conjugate_gradient_covariance")
                    v = ModifiedCG(self.K, vector.reshape(-1, 1), iterations=self.cg_iterations_covariance, tol=self.cg_tol,
                                            use_p_cholesky_preconditioner=True,
                                            preconditioner=self.preconditioner, reorthogonalization = self.reorthogonalization)  # , cg_iterations = self.cg_iterations,, printbar=True
                    set_global_section("other_predict")
                    v = np_dot(v.T, K_test_train[prediction_counter, :])

                elif self.cholesky:
                    set_global_section("section_cholesky_covariance")
                    v = forward_substitution(self.L, vector).reshape(1,-1)
                    set_global_section("other_predict")
                    v = np_dot(v, v.T)

                elif self.direct_inverse_cg:
                    set_global_section("other_predict")
                    vector = vector.reshape(1,-1)
                    v = np_dot(np_dot(vector, self.K_inv), vector.T)
                elif self.vanilla_cg:
                    set_global_section("section_conjugate_gradient_covariance")
                    v = ModifiedCG(self.K, vector.reshape(-1, 1), tol=self.cg_tol, iterations=self.cg_iterations_covariance, reorthogonalization = self.reorthogonalization)
                    set_global_section("other_predict")
                    v = np_dot(v.T, K_test_train[prediction_counter, :])

                else:
                    raise Exception("Sorry, incorrect configuration file")

                append_later.append(v)

            V = np.array(append_later).reshape(-1, 1)
            set_global_section("section_rbf_kernel_X*X*")
            K_test_test = rbf_kernel(testpoint_x, self.length_scales, self.noise, self.output_scale, fit=True)
            set_global_section("other_predict")
            if testset:
                self.y_cov_test.append(np_subtract(K_test_test.flatten()[0], V)[0][0])
            else:
                self.y_cov_train.append(np_subtract(K_test_test.flatten()[0], V)[0][0])

        if testset:
            self.y_mean_test = np.array(self.y_mean_test)
            self.y_cov_test = np.array(
                list(map(lambda x: sqrt_float(np_max(np.array([gmpy2.mpfr("0"), x]))), self.y_cov_test)))
            return self.y_mean_test, self.y_cov_test

        else:
            self.y_mean_train = np.array(self.y_mean_train)
            self.y_cov_train = np.array(
                list(map(lambda x: sqrt_float(np_max(np.array([gmpy2.mpfr("0"), x]))), self.y_cov_train)))
            count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state

            return self.y_mean_train, self.y_cov_train


    def eval(self):
        count_enabler.counting_enabled.old_state = copy.copy(count_enabler.counting_enabled.condition)
        count_enabler.counting_enabled.condition = False
        #pretty_print_sections_info()

        rms_flexGP_test = mean_squared_error(self.y_mean_test, self.y_test.astype(np.double), squared=False)
        rms_flexGP_train = mean_squared_error(self.y_mean_train, self.y_train.astype(np.double), squared=False)

        covDiff_flexGP_train = SumOfDifferencesOfConfidenceInterval(y_true= self.y_train, y_predictions=self.y_mean_train, y_predictions_variances=self.y_cov_train, target="FlexGPLearning")
        metric_covDiff_flexGP_train = sum_abs(covDiff_flexGP_train)
        covDiff_flexGP_test = SumOfDifferencesOfConfidenceInterval(y_true= self.y_test, y_predictions=self.y_mean_test, y_predictions_variances=self.y_cov_test, target="FlexGPGeneralization")
        metric_covDiff_flexGP_test = sum_abs(covDiff_flexGP_test)

        covDiff_scikit_train = SumOfDifferencesOfConfidenceInterval(y_true= self.y_train, y_predictions=self.scikit_mean_train, y_predictions_variances=self.scikit_std_train, target="BaselineLearning")
        metric_covDiff_scikit_train = sum_abs(covDiff_scikit_train)
        covDiff_scikit_test = SumOfDifferencesOfConfidenceInterval(y_true= self.y_test, y_predictions=self.scikit_mean_test, y_predictions_variances=self.scikit_std_test, target="BaselineGeneralization")
        metric_covDiff_scikit_test = sum_abs(covDiff_scikit_test)

        print("RMS Train FlexGP")
        print(rms_flexGP_train)

        #print("CovDiff Train FlexGP")
        #print(metric_covDiff_flexGP_train)

        print("RMS Test FlexGP")
        print(rms_flexGP_test)

        print("CovDiff Test FlexGP")
        print(metric_covDiff_flexGP_test)

        print("RMS Train ScikitChol")
        print(self.rms_scikit_test)

        #print("CovDiff Train ScikitChol")
        #print(metric_covDiff_scikit_train)

        print("RMS Test ScikitChol")
        print(self.rms_scikit_test)

        print("CovDiff Test ScikitChol")
        print(metric_covDiff_scikit_test)

        print("Scikit Hyperparameters")
        print(self.gpr.get_params())
        print(self.gpr.kernel_)
        rms_flexGP_test = mean_squared_error(self.y_mean_test, self.y_test.astype(np.double), squared=False)
        rms_flexGP_train = mean_squared_error(self.y_mean_train, self.y_train.astype(np.double), squared=False)

        #print(covDiff_flexGP_train)
        #print(covDiff_flexGP_test)

        #print(covDiff_scikit_train)
        #print(covDiff_scikit_test)

        counts_info_to_csv(config.experiment.config["name"], averaged=False)

        average_testpoint_section(len(self.y_test))

        counts_info_to_csv(config.experiment.config["name"], averaged=True)

        pretty_print_sections_info_to_csv(config.experiment.config["name"], rms_flexGP_train, rms_flexGP_test, self.rms_scikit_train, self.rms_scikit_test, metric_covDiff_flexGP_train, metric_covDiff_flexGP_test, metric_covDiff_scikit_train, metric_covDiff_scikit_test, covDiff_flexGP_train, covDiff_scikit_train,  covDiff_scikit_test, covDiff_flexGP_test)
        #pretty_print_sections_info()

        reset_wrapped_functions()
        count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state
        return
    def failed(self, message):
        output_folder = "experiments/results/" + config.experiment.config["name"] + "/"
        os.makedirs(output_folder, exist_ok=True)
        with open(output_folder + "failed.ignore", "w", newline='') as ignoreFile:
            ignoreFile.write(message.format_exc())
        shutil.copy2("experiments/" + config.experiment.config["name"] + ".config", output_folder)  # target filename is /dst/dir/file.ext
        reset_wrapped_functions()
        count_enabler.counting_enabled.condition = count_enabler.counting_enabled.old_state

def SumOfDifferencesOfConfidenceInterval(y_true, y_predictions, y_predictions_variances, target):
    directory = "experiments/results/" + config.experiment.config["name"]
    filename = directory + "/" + config.experiment.config["name"] + "_" + target + "_covdiff.csv"
    os.makedirs(directory, exist_ok=True)

    with open(filename, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["confidence level", "predicted std", "true value", "upper confidence bound", "lower confidence bound"])

        differences = []
        for confidence_level in np.arange (0.0,1.0,0.01):
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            z_score = round(z_score, 2)
            # lower bound y_predictions + z_score * y_predictions_variances
            # upper bound y_predictions - z_score * y_predictions_variances

            rows = [(y_true,variance, y_prediction + z_score * variance,y_prediction - z_score * variance) for y_prediction, variance, y_true in zip(y_predictions, y_predictions_variances, y_true)]
            for true, variance, up, low in rows:
                csv_writer.writerow([confidence_level,variance, true, up, low ])
            confidence_interval_absolute_hits = sum([y_true < (y_prediction + z_score * variance) and y_true > (y_prediction - z_score * variance) for y_prediction, variance, y_true in zip(y_predictions, y_predictions_variances,y_true)])
            percentage_interval_hits = confidence_interval_absolute_hits/len(y_predictions)
            difference = percentage_interval_hits-confidence_level
            differences.append(*difference)
        return differences

def sum_abs(input):
    return sum(abs(x) for x in input)/len(input)*2