from enum import Enum, auto

class GridPropertyName(Enum):
    """All the available property names for disk properties."""
    
    DISK_RADIUS = auto()
    INCLINATION = auto()
    TILT = auto()
    FX_MAP = auto()
    FY_MAP = auto()
    DIAGNOSTIC_MAP = auto()
    GRADIENT = auto()
    GRADIENT_FIT = auto()

    def __str__(self) -> str:
        """
        This method serves as the print value for this class

        Returns
        -------
        disk_property_name : str
            String value of the enum selected.
        """
        return self.get_name()

    def get_name(self) -> str:
        """
        This method is used to parse the name in a pretty way.

        Returns
        -------
        name : str
            Human readable version of enum name.
        """
        words = self.name.split('_')
        formatted_words = []
        
        for word in words:
            formatted_words.append(word.capitalize())
        
        name = ' '.join(formatted_words)

        return name