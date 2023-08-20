package csc_484;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Scanner;

// The 4 attributes in each data point (all binary)
enum Attribute {
    COINS_IN_ROOM, CLOSE_TO_CHAR, FIVE_COINS_SINCE, FIVE_SECOND_ROOM
};

// Each data point has exactly one action
enum Action {
    CHASE, COLLECT_COIN, ROAM_ROOM, CHANGE_ROOM
};

public class DecisionTreeLearning {

    private static class DataPoint {
        HashMap<Attribute, Boolean> attributeValues;
        Action                      action;

        DataPoint ( final boolean coinsInRoom, final boolean closeToChar, final boolean fiveCoinsSince,
                final boolean fiveSecondRoom, final Action action ) {
            attributeValues = new HashMap<Attribute, Boolean>( 4 );
            attributeValues.put( Attribute.COINS_IN_ROOM, coinsInRoom );
            attributeValues.put( Attribute.CLOSE_TO_CHAR, closeToChar );
            attributeValues.put( Attribute.FIVE_COINS_SINCE, fiveCoinsSince );
            attributeValues.put( Attribute.FIVE_SECOND_ROOM, fiveSecondRoom );
            this.action = action;
        }

        DataPoint ( final String line ) {
            attributeValues = new HashMap<Attribute, Boolean>( 4 );
            final Scanner s = new Scanner( line );
            attributeValues.put( Attribute.COINS_IN_ROOM, s.nextInt() == 1 );
            attributeValues.put( Attribute.CLOSE_TO_CHAR, s.nextInt() == 1 );
            attributeValues.put( Attribute.FIVE_COINS_SINCE, s.nextInt() == 1 );
            attributeValues.put( Attribute.FIVE_SECOND_ROOM, s.nextInt() == 1 );

            switch ( s.next() ) {
                case "Chase":
                    action = Action.CHASE;
                    break;
                case "CollectNearestCoin":
                    action = Action.COLLECT_COIN;
                    break;
                case "RoamRoom":
                    action = Action.ROAM_ROOM;
                    break;
                case "ChangeRoom":
                    action = Action.CHANGE_ROOM;
                    break;
            }
            s.close();
        }
    }

    private static class DataSet {
        public final ArrayList<DataPoint> data;
        public final HashSet<Attribute>   filteredOn;

        DataSet () {
            data = new ArrayList<DataPoint>();
            filteredOn = new HashSet<Attribute>();
        }

        /**
         * @return true if all data points have the same action
         */
        public boolean allSameAction () {
            final Action firstAction = data.get( 0 ).action;
            for ( final DataPoint d : data ) {
                if ( d.action != firstAction ) {
                    return false;
                }
            }
            return true;
        }

        public DataPoint get ( final int i ) {
            return data.get( i );
        }

        public int size () {
            return data.size();
        }

        /**
         * Return a subset of this data where every data point must have the
         * given attribute equal to given boolean value
         *
         * @param filterAttribute
         *            the attribute in consideration
         * @param value
         *            the value the condition must have to be included in the
         *            return
         * @return A subset of this graph where all filterAttribute == value
         */
        public DataSet filter ( final Attribute filterAttribute, final boolean value ) {
            final DataSet filtered = new DataSet();
            for ( final DataPoint d : data ) {
                if ( d.attributeValues.get( filterAttribute ) == value ) {
                    filtered.data.add( d );
                }
            }
            for ( final Attribute f : this.filteredOn ) {
                filtered.filteredOn.add( f );
            }
            filtered.filteredOn.add( filterAttribute );
            return filtered;
        }

        /**
         * Returns the proportion of the data set whose attribute = value
         */
        public double proportionOfAttribute ( final Attribute attribute, final boolean value ) {
            if ( data.size() == 0 ) {
                return 0;
            }
            int attributeCount = 0;
            for ( final DataPoint d : data ) {
                if ( d.attributeValues.get( attribute ) == value ) {
                    attributeCount++;
                }
            }
            return (double) attributeCount / (double) data.size();
        }

        /**
         * Returns the proportion of the data set that has the given action
         */
        public double proportionOfAction ( final Action action ) {
            if ( data.size() == 0 ) {
                return 0;
            }
            int actionCount = 0;
            for ( final DataPoint d : data ) {
                if ( d.action == action ) {
                    actionCount++;
                }
            }
            return (double) actionCount / (double) data.size();
        }

        /**
         * Creates an action node from this sample of data where each action has
         * probability equal to its proportion in the DataSet
         */
        public ActionNode randomizeRemainingActions () {

            final double chaseP = proportionOfAction( Action.CHASE );
            final double collectP = proportionOfAction( Action.COLLECT_COIN );
            final double roamP = proportionOfAction( Action.ROAM_ROOM );
            final double changeP = proportionOfAction( Action.CHANGE_ROOM );

            final ActionNode node = new ActionNode();
            if ( chaseP > 0 ) {
                node.actions.put( Action.CHASE, chaseP );
                node.rawActions.add( Action.CHASE );
            }
            if ( collectP > 0 ) {
                node.actions.put( Action.COLLECT_COIN, collectP );
                node.rawActions.add( Action.COLLECT_COIN );
            }
            if ( roamP > 0 ) {
                node.actions.put( Action.ROAM_ROOM, roamP );
                node.rawActions.add( Action.ROAM_ROOM );
            }
            if ( changeP > 0 ) {
                node.actions.put( Action.CHANGE_ROOM, changeP );
                node.rawActions.add( Action.CHANGE_ROOM );
            }
            if ( node.actions.size() > 1 ) {
                node.isRandom = true;
            }
            return node;

        }

        public double entropy () {
            double entropy = 0;
            final double changeProportion = proportionOfAction( Action.CHANGE_ROOM );
            if ( changeProportion != 0 ) {
                entropy -= changeProportion * Math.log10( changeProportion );
            }

            final double chaseProportion = proportionOfAction( Action.CHASE );
            if ( chaseProportion != 0 ) {
                entropy -= chaseProportion * Math.log10( chaseProportion );
            }

            final double collectProportion = proportionOfAction( Action.COLLECT_COIN );
            if ( collectProportion != 0 ) {
                entropy -= collectProportion * Math.log10( collectProportion );
            }

            final double roamProportion = proportionOfAction( Action.ROAM_ROOM );
            if ( roamProportion != 0 ) {
                entropy -= roamProportion * Math.log10( roamProportion );
            }
            return entropy;
        }

        public double informationGain ( final Attribute attribute ) {
            double startEntropy = entropy();
            startEntropy -= proportionOfAttribute( attribute, true ) * filter( attribute, true ).entropy();
            startEntropy -= proportionOfAttribute( attribute, false ) * filter( attribute, false ).entropy();
            return startEntropy;
        }

        /**
         * The best criteria to split the data set with
         */
        public Attribute bestCriteria () {

            double maxInformationGain = -1 * Double.MIN_VALUE;
            Attribute maximizingAttribute = null;

            if ( !filteredOn.contains( Attribute.CLOSE_TO_CHAR ) ) {
                final double thisInformationGain = this.informationGain( Attribute.CLOSE_TO_CHAR );
                if ( thisInformationGain > maxInformationGain ) {
                    maxInformationGain = thisInformationGain;
                    maximizingAttribute = Attribute.CLOSE_TO_CHAR;
                }
            }
            if ( !filteredOn.contains( Attribute.COINS_IN_ROOM ) ) {
                final double thisInformationGain = this.informationGain( Attribute.COINS_IN_ROOM );
                if ( thisInformationGain > maxInformationGain ) {
                    maxInformationGain = thisInformationGain;
                    maximizingAttribute = Attribute.COINS_IN_ROOM;
                }
            }
            if ( !filteredOn.contains( Attribute.FIVE_COINS_SINCE ) ) {
                final double thisInformationGain = this.informationGain( Attribute.FIVE_COINS_SINCE );
                if ( thisInformationGain > maxInformationGain ) {
                    maxInformationGain = thisInformationGain;
                    maximizingAttribute = Attribute.FIVE_COINS_SINCE;
                }
            }
            if ( !filteredOn.contains( Attribute.FIVE_SECOND_ROOM ) ) {
                final double thisInformationGain = this.informationGain( Attribute.FIVE_SECOND_ROOM );
                if ( thisInformationGain > maxInformationGain ) {
                    maxInformationGain = thisInformationGain;
                    maximizingAttribute = Attribute.FIVE_SECOND_ROOM;
                }
            }

            if ( maximizingAttribute == null ) {
                System.exit( 1 );
            }
            return maximizingAttribute;
        }
    }

    private static abstract class Node {
        boolean isDecision;
        Node    falseChild;
        Node    trueChild;

        @Override
        public abstract String toString ();
    }

    private static class ActionNode extends Node {
        // stores actions w/o probability
        public ArrayList<Action>       rawActions;

        // stores actions w/ probability
        public HashMap<Action, Double> actions;

        // true if node has more than one action
        boolean                        isRandom;

        /**
         * Creates an Action Node with the given possible actions[i] and
         * probabilities[i]
         *
         * @param n
         *            number of actions
         */
        ActionNode ( final Action[] actions, final double[] probabilities, final int n ) {
            this.rawActions = new ArrayList<Action>();
            this.actions = new HashMap<Action, Double>( n );
            for ( int i = 0; i < n; i++ ) {
                this.actions.put( actions[i], probabilities[i] );
                this.rawActions.add( actions[i] );
            }
            this.isRandom = n > 1;
            this.isDecision = false;
            falseChild = null;
            trueChild = null;
        }

        ActionNode () {
            this.rawActions = new ArrayList<Action>();
            this.actions = new HashMap<Action, Double>( 4 );
            this.isDecision = false;
            falseChild = null;
            trueChild = null;
        }

        @Override
        public String toString () {
            if ( isRandom ) {
                final StringBuilder s = new StringBuilder( "{" );
                for ( final Entry<Action, Double> entry : actions.entrySet() ) {
                    s.append( stringOf( entry.getKey() ) );
                    s.append( " : " );
                    s.append( String.format( "%.3f", entry.getValue() ) );
                    s.append( ", " );
                }
                s.deleteCharAt( s.length() - 1 );
                s.deleteCharAt( s.length() - 1 ); // deletes last comma
                s.append( "}" );
                return s.toString();
            }
            else {
                return "{" + stringOf( rawActions.get( 0 ) ) + "}";
            }
        }
    }

    /**
     * Represents a node in the tree where an attribute is checked All Decision
     * Nodes are binary
     */
    private static class DecisionNode extends Node {
        Attribute attribute;

        DecisionNode ( final Attribute attribute ) {
            this.isDecision = true;
            this.attribute = attribute;
            falseChild = null;
            trueChild = null;
        }

        @Override
        public String toString () {
            return "(" + stringOf( attribute ) + "?)";
        }
    }

    private static DataSet loadDataSet () throws Exception {
        final File f = new File( "C:\\Users\\yousi\\data.txt" );
        final Scanner scanner = new Scanner( f );

        final DataSet fileSet = new DataSet();
        while ( scanner.hasNextLine() ) {
            fileSet.data.add( new DataPoint( scanner.nextLine() ) );
        }
        return fileSet;
    }

    /**
     * Recursively builds a decision tree from the data set using ID3
     *
     * @param samples
     *            the data to build a root node with
     * @return the root node of the decision tree
     */
    private static Node buildDecisionTree ( final DataSet samples ) {
        // if there is no data for this configuration of attributes, do a
        // uniformly random action
        if ( samples.size() == 0 ) {
            final Action[] actions = { Action.CHASE, Action.COLLECT_COIN, Action.ROAM_ROOM, Action.CHANGE_ROOM };
            final double[] probabilities = { 0.25, 0.25, 0.25, 0.25 };
            return new ActionNode( actions, probabilities, 4 );
        }

        // If all attributes have already been considered, randomize the
        // remaining actions based on their proprotions
        else if ( samples.filteredOn.size() == 4 ) {
            return samples.randomizeRemainingActions();
        }

        // if all of the data points to one action, do that action
        else if ( samples.allSameAction() ) {
            final Action[] actions = { samples.get( 0 ).action };
            final double[] probabilities = { 1 };
            return new ActionNode( actions, probabilities, 1 );
        }

        // Otherwise, recursively build the tree by splitting on true/false
        // of
        // the attribute with the most information gain
        else {
            final Attribute conditionAttribute = samples.bestCriteria();
            final DecisionNode node = new DecisionNode( conditionAttribute );
            final DataSet falseSamples = samples.filter( conditionAttribute, false );
            final DataSet trueSamples = samples.filter( conditionAttribute, true );
            node.falseChild = buildDecisionTree( falseSamples );
            node.trueChild = buildDecisionTree( trueSamples );
            return node;
        }
    }

    private static String stringOf ( final Action action ) {
        switch ( action ) {
            case CHANGE_ROOM:
                return "Change Room";
            case CHASE:
                return "Chase";
            case COLLECT_COIN:
                return "Collect Coin";
            case ROAM_ROOM:
                return "Roam Room";
            default:
                return "ERROR";
        }
    }

    private static String stringOf ( final Attribute attribute ) {
        switch ( attribute ) {
            case CLOSE_TO_CHAR:
                return "Close to Char";
            case COINS_IN_ROOM:
                return "Coins In Room";
            case FIVE_COINS_SINCE:
                return "Five Coins Since Encounter";
            case FIVE_SECOND_ROOM:
                return "Five Seconds In Room";
            default:
                return "ERROR";
        }
    }

    /**
     * @Source https://www.baeldung.com/java-print-binary-tree-diagram
     */
    private static void traverseNodes ( final StringBuilder sb, final String padding, final String pointer,
            final Node node, final boolean hasRightSibling ) {
        if ( node != null ) {
            sb.append( "\n" );
            sb.append( padding );
            sb.append( pointer );
            sb.append( node.toString() );

            final StringBuilder paddingBuilder = new StringBuilder( padding );
            if ( hasRightSibling ) {
                paddingBuilder.append( "│  " );
            }
            else {
                paddingBuilder.append( "   " );
            }

            final String paddingForBoth = paddingBuilder.toString();
            final String pointerRight = "└──T:";
            final String pointerLeft = ( node.trueChild != null ) ? "├──F:" : "└──F:";

            traverseNodes( sb, paddingForBoth, pointerLeft, node.falseChild, node.trueChild != null );
            traverseNodes( sb, paddingForBoth, pointerRight, node.trueChild, false );
        }
    }

    /**
     * @Source https://www.baeldung.com/java-print-binary-tree-diagram
     */
    private static String preOrderTraverse ( final Node root ) {

        if ( root == null ) {
            return "";
        }

        final StringBuilder sb = new StringBuilder();
        sb.append( root.toString() );

        final String pointerRight = "└──T:";
        final String pointerLeft = ( root.trueChild != null ) ? "├──F:" : "└──F:";

        traverseNodes( sb, "", pointerLeft, root.falseChild, root.trueChild != null );
        traverseNodes( sb, "", pointerRight, root.trueChild, false );

        return sb.toString();

    }

    /**
     * @Source https://www.baeldung.com/java-print-binary-tree-diagram
     */
    private static void printTree ( final Node root ) {
        System.out.print( preOrderTraverse( root ) );
    }

    public static void main ( final String[] args ) {

        DataSet dataSamples = null;
        try {
            dataSamples = loadDataSet();
        }
        catch ( final Exception e ) {
            System.out.println( "err" );
            System.exit( 1 );
        }

        final Node root = buildDecisionTree( dataSamples );
        printTree( root );

    }
}
