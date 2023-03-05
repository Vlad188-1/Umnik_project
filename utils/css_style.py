from dataclasses import dataclass

@dataclass
class Constants:
    PROCESSING_BUTTONS_STYLES = """
        QWidget {background-color:lightgrey}
        QPushButton {background-color:white; border: 0px solid red;}
        QPushButton:hover
        {
            background-color:rgb(0,130,150);
            border:2px solid #5F92B2;
            border-radius:5px;
            color:white;
        }
        QPushButton:hover:pressed
        {
            background-color:rgb(50, 205, 50); 
            border:2px solid #3C80B1;
            border-radius:5px;
            color:white;
        }
        """
    UPDATE_TABLE_BUTTON_STYLE = """
        QPushButton {background-color:lightgreen; border: 1px solid black; border-radius:5px; font: bold}
        QPushButton:hover
        {
            background-color:orange;
        }
    """
    SELECTALL_BUTTON = """
        QPushButton {background-color:beige; border: 1px solid black; border-radius:5px; font: bold}
        QPushButton:hover
        {
            background-color:cyan;
        }
    """
    DELETE_NAN = """
        QPushButton {background-color:lightblue; border: 1px solid black; border-radius:5px; font: bold}
        QPushButton:hover
        {
            background-color:cyan;
        }
    """
    # SETTING_MODEL= """
    # QLabel {font-size: 18pt;}
    # """