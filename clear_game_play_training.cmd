REM delete the directory base_agent.pkl file as well as base_agent_training_context.pkl as well as invalid_move_training_accuracy.log
rd /s /q .\model\tic_tac_toe_first_agent.pkl
rd /s /q .\model\tic_tac_toe_second_agent.pkl
del .\training_progress\training_context.pkl