import os
import sys

from pydrake.all import(LoadModelDirectives, Parser,
                        ProcessModelDirectives, MultibodyPlant)

package_xml_file = os.path.join(os.path.dirname(__file__), "models/package.xml")
models_folder = os.path.join(os.path.dirname(__file__), "models")

def GetParser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    ConfigureParser(parser)
    return parser

def ConfigureParser(parser):
    """Add the manipulation/package.xml index to the given Parser."""
    parser.package_map().AddPackageXml(filename=package_xml_file)
    AddPackagePaths(parser)

def AddPackagePaths(parser):
    parser.package_map().PopulateFromFolder(str(models_folder))