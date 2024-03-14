case "$1" in
    "test1")
        echo "Starting test1"
        python main.py --uselinear 1 --usefunc 0
        ;;
    "test2")
        echo "Starting test2"
        python main.py --uselinear 0 --usefunc 1
        ;;
    *)
        echo "Starting test1"
        python main.py --uselinear 1 --usefunc 0
        ;;
esac