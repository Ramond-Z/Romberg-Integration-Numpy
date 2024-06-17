import numpy as np
import matplotlib.pyplot as plt
import timeit


from typing import Callable, Dict, List, Optional


def RombergIntegration(f: Callable[[np.ndarray], np.ndarray], sequence_length: int, x_start: float, x_end: float) -> Dict[str, np.ndarray]:
    """
    Compute the Romberg integral sequence for a given function over a specified interval.

    Parameters:
    - f (Callable[[np.ndarray], np.ndarray]): The function to integrate. It should accept a numpy array and return a numpy array.
    - sequence_length (int): The length of the Romberg sequence, indicating the depth of refinement.
    - x_start (float): The start of the integration interval.
    - x_end (float): The end of the integration interval.

    Returns:
    - Dict[str, np.ndarray]: A dictionary containing the sequences for Trapezoidal, Simpson, Cotes, and Romberg integrals.
    """

    assert isinstance(
        sequence_length, int) and sequence_length > 0, "sequence_length must be a positive integer"
    assert x_end > x_start, "x_end must be greater than x_start"

    partition_points: List[np.ndarray] = [np.linspace(
        x_start, x_end, 2**i, dtype=np.double) for i in range(1, sequence_length + 4)]

    trapezoidal_sequence: np.ndarray = np.array(
        [TrapezoidalIntegration(f, x_start, x_end, partition) for partition in partition_points], dtype=np.double
    )

    simpson_sequence: np.ndarray = Extrapolate(
        trapezoidal_sequence[:-1], trapezoidal_sequence[1:], -1, 4)
    cotes_sequence: np.ndarray = Extrapolate(
        simpson_sequence[:-1], simpson_sequence[1:], -1, 16)
    romberg_sequence: np.ndarray = Extrapolate(
        cotes_sequence[:-1], cotes_sequence[1:], -1, 64)

    return {
        'Trapezoidal': trapezoidal_sequence,
        'Simpson': simpson_sequence,
        'Cotes': cotes_sequence,
        'Romberg': romberg_sequence,
    }


def TrapezoidalIntegration(f: Callable[[np.ndarray], np.ndarray], x_start: float, x_end: float, partition_points: np.ndarray) -> float:
    """
    Compute the integral of a function using the Trapezoidal rule over a given interval with specified partition points.

    Parameters:
    - f (Callable[[np.ndarray], np.ndarray]): The function to integrate.
    - x_start (float): The start of the integration interval.
    - x_end (float): The end of the integration interval.
    - partition_points (np.ndarray): The array of partition points.

    Returns:
    - float: The computed integral using the Trapezoidal rule.
    """

    assert x_end > x_start, "x_end must be greater than x_start"

    h = (x_end - x_start) / (len(partition_points) - 1)
    nodes = f(partition_points)
    return h / 2 * (2 * nodes.sum() - nodes[0] - nodes[-1])


def Extrapolate(x1: np.ndarray, x2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """
    Perform Richardson extrapolation on two sequences.

    Parameters:
    - x1 (np.ndarray): The first sequence.
    - x2 (np.ndarray): The second sequence.
    - w1 (float): The weight for the first sequence.
    - w2 (float): The weight for the second sequence.

    Returns:
    - np.ndarray: The extrapolated sequence.
    """

    assert w1 + w2 != 0, "The sum of weights w1 and w2 must not be zero"

    return (w1 * x1 + w2 * x2) / (w1 + w2)


def example_function(x: np.ndarray) -> np.ndarray:
    """
    Example function to integrate: f(x) = (x^2 + x + 1) * cos(x).

    Parameters:
    - x (np.ndarray): The input values.

    Returns:
    - np.ndarray: The computed function values.
    """

    return (x**2 + x + 1) * np.cos(x)


def ShowResults(results: Dict[str, np.ndarray], ground_truth: float = None) -> None:
    """
    Display the results of the Romberg integral sequence using a plot.

    Parameters:
    - results (Dict[str, np.ndarray]): The computed sequences from Romberg integration.
    - ground_truth (float, optional): The ground truth value for the integral, used for error calculation. Default is None.
    """

    T, S, C, R = results.values()

    # Plot each sequence
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(np.arange(len(T)), np.abs(T - ground_truth) if ground_truth is not None else T,
             label='Trapezoidal Sequence', marker='o')
    plt.plot(np.arange(len(S)), np.abs(S - ground_truth) if ground_truth is not None else S,
             label='Simpson Sequence', marker='s')
    plt.plot(np.arange(len(C)), np.abs(C - ground_truth) if ground_truth is not None else C,
             label='Cotes Sequence', marker='^')
    plt.plot(np.arange(len(R)), np.abs(R - ground_truth) if ground_truth is not None else R,
             label='Romberg Sequence', marker='d')

    # Set plot limits and labels
    # plt.ylim(-.2, .2)
    plt.legend()
    plt.title('Romberg Integral')
    plt.xlabel('Iteration')
    plt.ylabel('Integral Value' if ground_truth is None else 'Integral Error')
    plt.show()


def SaveResults(results: Dict[str, np.ndarray], path: str, gt: Optional[float] = None) -> None:
    """
    Save the Romberg integration results to a Markdown formatted table.

    Parameters:
    - results (Dict[str, np.ndarray]): The computed sequences from Romberg integration.
    - path (str): The file path where the table will be saved.
    - gt (Optional[float]): The ground truth value for the integral, used for error calculation. Default is None.
    """

    trapezoidal_sequence = results.get('Trapezoidal', [])
    simpson_sequence = results.get('Simpson', [])
    cotes_sequence = results.get('Cotes', [])
    romberg_sequence = results.get('Romberg', [])

    max_length = max(len(trapezoidal_sequence), len(simpson_sequence), len(cotes_sequence), len(romberg_sequence))
    rows = ["| Trapezoidal Sequence | Trapezoidal Error | Simpson Sequence | Simpson Error | Cotes Sequence | Cotes Error | Romberg Sequence | Romberg Error |", "|---|---|---|---|---|---|---|---|"]

    for i in range(max_length):
        t_val = trapezoidal_sequence[i] if i < len(trapezoidal_sequence) else None
        t_err = abs(t_val - gt) if gt is not None and t_val is not None else None
        s_val = simpson_sequence[i] if i < len(simpson_sequence) else None
        s_err = abs(s_val - gt) if gt is not None and s_val is not None else None
        c_val = cotes_sequence[i] if i < len(cotes_sequence) else None
        c_err = abs(c_val - gt) if gt is not None and c_val is not None else None
        r_val = romberg_sequence[i] if i < len(romberg_sequence) else None
        r_err = abs(r_val - gt) if gt is not None and r_val is not None else None
        row = f"| {t_val} | {t_err} | {s_val} | {s_err} | {c_val} | {c_err} | {r_val} | {r_err} |"
        row = row.replace("None", " ")
        rows.append(row)

    with open(path, 'w') as file:
        file.write("\n".join(rows))


def TestRunTime(n_iter: int) -> float:
    """
    Measure the average execution time of RombergIntegration function over `n_iter` iterations.

    Parameters:
    - n_iter (int): Number of iterations to measure the execution time.

    Returns:
    - float: Average execution time per iteration in seconds.
    """
    func: Callable[[], dict] = lambda: RombergIntegration(example_function, 15, 0, np.pi / 2)
    execute_time: float = timeit.timeit(func, number=n_iter)
    return execute_time / n_iter


if __name__ == '__main__':

    # Example usage of the Romberg integration function

    results = RombergIntegration(example_function, 15, 0, np.pi / 2)
    gt = 2.038197427067
    ShowResults(results, ground_truth=gt)
    SaveResults(results, 'results.md', gt)

    # method used for examine average run time

    average_run_time = TestRunTime(100)
    print('Average run time: {:.6f} seconds'.format(average_run_time))
