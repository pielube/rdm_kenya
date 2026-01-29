"""Run the RDM workflow for OSeMOSYS scenarios."""
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from workflow import z_auxiliar_code as AUX

start = time.time()

WORKFLOW_DIR = Path("workflow")
SCENARIOS_DIR = WORKFLOW_DIR / "0_Scenarios"
EXECUTABLES_DIR = WORKFLOW_DIR / "1_Experiment" / "Executables"
FUTURES_DIR = WORKFLOW_DIR / "1_Experiment" / "Experimental_Platform" / "Futures"
MODEL_STRUCTURE_PATH = (WORKFLOW_DIR / "1_Experiment" / "0_From_Confection" / "B1_Model_Structure.xlsx")
OSEMOSYS_STRUCTURE_PATH = WORKFLOW_DIR / "2_Miscellaneous" / "OSeMOSYS_Structure.xlsx"
SHAPE_FILE_PATH = WORKFLOW_DIR / "2_Miscellaneous" / "shape_of_demand.csv"

scenario_files = os.listdir(SCENARIOS_DIR)

interface_dir = Path("interface_rdm_inputs")

setup_table = pd.read_csv(interface_dir / "Setup.csv")

num_time_slices_sdp = int(setup_table.loc[0, "Timeslices_model"])
run_base_future = str(setup_table.loc[0, "Run_Base_Future"])
run_rdm = str(setup_table.loc[0, "Run_RDM"])
solver = str(setup_table.loc[0, "Solver"])
osemosys_model = str(setup_table.loc[0, "OSeMOSYS_Model_Name"])

parameters_to_print = pd.read_csv(interface_dir / "To_Print.csv")

if run_base_future == "Yes":
    # Step 1: Delete ResultsPath param
    for scenario in scenario_files:
        scenario_path = SCENARIOS_DIR / scenario
        with open(scenario_path, "r") as fp:
            lines = fp.readlines()
        with open(scenario_path, "w") as fp:
            for line in lines:
                if "ResultsPath" not in line:
                    fp.write(line)

    print("Step 1 finished")

    # Step 2: Clean folders in ./workflow/1_Experiment/0_From_Confection/
    target_dir = WORKFLOW_DIR / "1_Experiment" / "0_From_Confection"
    for files in os.listdir(target_dir):
        path = target_dir / files
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

    print("Step 2 finished")

    # Step 3: Obtain ModelStructure & DefaultParams
    # Note: all scenarios must have the same sets. We only use one of the scenario
    # files to obtain the model structure and default parameters.
    AUX.obtain_structure_file(
        str(SCENARIOS_DIR / scenario_files[0]),
        str(MODEL_STRUCTURE_PATH),
        str(OSEMOSYS_STRUCTURE_PATH),
        num_time_slices_sdp,
    )

    print("Step 3 finished")

    # Step 4: Clean ./workflow/1_Experiment/Executables folder except .py file
    # Clean folders
    target_dir = EXECUTABLES_DIR
    for files in os.listdir(target_dir):
        path = target_dir / files
        if ".py" not in files:
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

    print("Step 4 finished")

    # Step 5: Clean ./workflow/1_Experiment/Experimental_Platform/Futures folder except .py file
    # Clean folders
    target_dir = FUTURES_DIR
    for files in os.listdir(target_dir):
        path = target_dir / files
        if ".py" not in files:
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

    print("Step 5 finished")

    # Step 6: Create folders for future 0 and multiple futures
    # Create a folder for each scenario
    for scenario in scenario_files:
        scenario_stem = scenario.replace(".txt", "")
        newpath = EXECUTABLES_DIR / f"{scenario_stem}_0"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    # Create a folder for each scenario
    for scenario in scenario_files:
        newpath = FUTURES_DIR / scenario.replace(".txt", "")
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    print("Step 6 finished")

    # Step 7: Paste scenarios future 0 TXT files in ./workflow/1_Experiment/Executables/
    for scenario in scenario_files:
        scenario_stem = scenario.replace(".txt", "")
        source_folder = str(SCENARIOS_DIR) + "/"
        destination_folder = str(EXECUTABLES_DIR / f"{scenario_stem}_0") + "/"
        # construct full file path
        source = source_folder + scenario
        destination = destination_folder + scenario_stem + "_0.txt"
        # copy files and write with timeslices quantity of the RDM_Interface.xlsx
        AUX.process_timeslices(source, num_time_slices_sdp, destination)
    print("Step 7 finished")

    # Step 8: Store data with scenarios data
    # Store data from executable file
    for scenario in scenario_files:
        scenario_stem = scenario.replace(".txt", "")

        # Isolate params in subfiles
        data_per_param, _ = AUX.isolate_params(str(SCENARIOS_DIR / scenario))

        # Generate CSV parameters files for each scenario
        list_dataframes, _, _ = AUX.generate_df_per_param(
            scenario_stem,
            data_per_param,
            num_time_slices_sdp,
        )

        # Create future 0 input dataset
        AUX.create_input_dataset_future_0(
            list_dataframes,
            scenario_stem,
            "./workflow/1_Experiment/Executables/" + scenario.replace(".txt", "_0/"),
        )

    print("Step 8 finished")

    # Step 9: Create future 0 output dataset
    # Output
    start1 = time.time()
    for scenario in scenario_files:
        scenario_stem = scenario.replace(".txt", "")
        # Run OSeMOSYS for each scenario
        AUX.run_osemosys(
            solver,
            "./workflow/1_Experiment/Executables/" + scenario.replace(".txt", "_0/"),
            "./workflow/1_Experiment/Executables/"
            + scenario.replace(".txt", "_0/")
            + scenario.replace(".txt", "_0.txt"),
            "./workflow/" + osemosys_model,
            "./workflow/1_Experiment/Executables/" + scenario.replace(".txt", "_0/") + scenario_stem,
        )

        print("Step 9.Input finished")

        print("Step 9.Output generated. Star long function")
        if solver in ("cbc", "cplex"):
            AUX.data_processor_new(
                "./workflow/1_Experiment/Executables/"
                + scenario.replace(".txt", "_0/")
                + scenario_stem
                + "_0_Output.sol",
                str(MODEL_STRUCTURE_PATH),
                scenario_stem,
                str(0),
                solver,
                parameters_to_print,
                "csv",
            )
        elif solver == "glpk":
            AUX.data_processor_new(
                "./workflow/1_Experiment/Executables/"
                + scenario.replace(".txt", "_0/")
                + scenario_stem
                + "_0_Output.txt",
                str(MODEL_STRUCTURE_PATH),
                scenario_stem,
                str(0),
                solver,
                parameters_to_print,
                "csv",
            )
        print("Step 9.Output finished")
    print("Step 9 finished")
    end_1 = time.time()
    time_elapsed_1 = int(round(end_1 - start1))
    print(
        "   The total time producing outputs and storing data of base futures have been: "
        + str(time_elapsed_1)
        + " seconds"
    )

if run_rdm == "Yes":
    # Step 10: Execute RDM experiment
    print("Start RDM experiment\n")
    AUX.run_scripts(
        "./workflow/1_Experiment/0_experiment_manager.py",
        solver,
        osemosys_model,
        os.path.abspath("interface_rdm_inputs"),
        shape_file=os.path.abspath(str(SHAPE_FILE_PATH)),
    )

    print("Step 10 finished\n")

    # Step 11: Execute RDM experiment
    start3 = time.time()
    print("Start Output Dataset Creator\n")
    AUX.run_scripts("./workflow/1_Experiment/1_output_dataset_creator.py")

    print("Step 11 finished\n")
    end_3 = time.time()
    time_elapsed_3 = int(round(end_3 - start3))
    print(
        "   The total time producing storing data of the experiment has been: "
        + str(time_elapsed_3)
        + " seconds"
    )

if solver == "cplex":
    for file in ["cplex.log", "clone1.log", "clone2.log"]:
        if os.path.exists(file):
            os.remove(file)

print("#####################################")
print("Processing completed successfully.")
print("#####################################")

end = time.time()
time_elapsed = int(round(end - start))
print("   The total time of the workflow has been: " + str(time_elapsed) + " seconds")
